#!/usr/bin/env python3

import os
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple, Any, Optional
import numpy as np

class ComprehensiveAnalyzer:

    def __init__(self):
        pass

    def _extract_answer_from_string(self, answer_str: str) -> str:

        if isinstance(answer_str, (int, float)):
            return str(answer_str)
        if isinstance(answer_str, str):

            match = re.search(r'answer:\s*(.+)', answer_str, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()

                if "," in extracted:
                    extracted = extracted.split(",")[0].strip()
                return extracted

            return answer_str.strip()
        return str(answer_str)

    def _analyze_gsm8k_format(self, data: Dict) -> Dict:

        test_results = data.get('test_results', [])
        consensus_results = data.get('consensus_results', [])

        agent_stats = data.get('agent_stats', {})

        malicious_agents = data.get('malicious_agents', [])
        if not malicious_agents and 'metadata' in data:
            malicious_agents = data['metadata'].get('malicious_agents', [])

        config = data.get('test_config', {}) or {}

        try:
            tr_root = data.get('test_results', {})
            if isinstance(tr_root, dict):
                if 'topology_type' not in config and 'topology_type' in tr_root:
                    config['topology_type'] = tr_root.get('topology_type')
                if 'topology' not in config and 'topology' in tr_root:
                    config['topology'] = tr_root.get('topology')

                if 'num_agents' not in config:
                    config['num_agents'] = tr_root.get('agent_count', tr_root.get('agents'))
                if 'num_malicious' not in config:
                    config['num_malicious'] = tr_root.get('malicious_count', tr_root.get('malicious'))
        except Exception:
            pass
        dataset_type = str(config.get('dataset_type', 'gsm8k')).lower()

        node_changes = {}

        if not agent_stats and test_results:

            all_agents = set()
            for result in test_results:
                individual_responses = result.get('individual_responses', [])
                for response in individual_responses:
                    all_agents.add(response.get('agent_id'))
            agent_stats = {aid: {} for aid in all_agents}

        agent_correct = {aid: 0 for aid in agent_stats}
        agent_total = {aid: 0 for aid in agent_stats}
        agent_final_correct = {aid: 0 for aid in agent_stats}             
        consensus_correct = 0
        total_rounds = len(test_results)

        for result in test_results:

            question_id = result.get('question_id', result.get('question', {}).get('question', f"question_{result.get('round_number', 1)}"))

            if dataset_type == 'safe':
                question_id = result.get('question', {}).get('question', question_id)
            else:

                if 'Jerry' in str(question_id):
                    question_id = 'gsm8k_jerry'
                elif len(str(question_id)) > 50:
                    question_id = f"gsm8k_{result.get('round_number', 1)}"

            agent_detailed = result.get('agent_detailed', {})
            individual_responses = result.get('individual_responses', [])

            if individual_responses:

                if dataset_type == 'safe':
                    def to_label(v: str) -> str:
                        s = str(v).strip().lower()
                        if s in ('1', 'safe'):
                            return 'safe'
                        return 'unsafe'
                else:
                    def to_label(v: str) -> str:

                        return str(v).strip()

                question_data = result.get('question', {})
                ce = result.get('consensus_evaluation', {}) or {}
                ca_raw = ce.get('correct_answer', question_data.get('answer'))

                if ca_raw is None:
                    ca_raw = question_data.get('correct_answer')

                if ca_raw is not None:
                    ca_norm = str(ca_raw).strip()
                    if dataset_type == 'safe':
                        _l = ca_norm.lower()
                        correct_answer = 'safe' if _l in ('1', 'safe') else 'unsafe'
                    else:
                        correct_answer = ca_norm
                else:
                    correct_answer = None

                for response in individual_responses:
                    agent_id = response.get('agent_id')
                    if not agent_id:
                        continue
                    if agent_id not in node_changes:
                        node_changes[agent_id] = []

                    md = response.get('metadata', {}) or {}

                    initial_raw = md.get('initial_answer', response.get('answer', ''))
                    final_raw = md.get('final_answer', response.get('answer', ''))

                    initial_val = to_label(str(initial_raw))
                    final_val = to_label(str(final_raw))
                    changed = (initial_val != final_val)

                    node_changes[agent_id].append({
                        'question': question_id,
                        'initial': initial_val,
                        'final': final_val,
                        'changed': changed
                    })

                    if correct_answer is not None:
                        if str(initial_val).strip().lower() == str(correct_answer).strip().lower():
                            agent_correct[agent_id] = agent_correct.get(agent_id, 0) + 1
                        if str(final_val).strip().lower() == str(correct_answer).strip().lower():
                            agent_final_correct[agent_id] = agent_final_correct.get(agent_id, 0) + 1
                        agent_total[agent_id] = agent_total.get(agent_id, 0) + 1

            elif agent_detailed:

                if dataset_type == 'safe':
                    def to_label(v: str) -> str:
                        s = str(v).strip().lower()
                        if s in ('1', 'safe'):
                            return 'safe'
                        return 'unsafe'
                else:
                    def to_label(v: str) -> str:
                        return str(v).strip()

                question_data = result.get('question', {})
                ce = result.get('consensus_evaluation', {}) or {}
                ca_raw = ce.get('correct_answer', question_data.get('answer', 'safe'))
                ca_norm = str(ca_raw).strip()
                if dataset_type == 'safe':
                    _l = ca_norm.lower()
                    correct_answer = 'safe' if _l in ('1', 'safe') else 'unsafe'
                else:
                    correct_answer = ca_norm

                initial_answers = {}
                consensus_answers = {}
                for agent_id, agent_data in agent_detailed.items():

                    raw_response = agent_data.get('raw_response', '')
                    if 'answer:' in raw_response:
                        init_token = raw_response.split('answer:')[1].split(',')[0].strip()
                    else:
                        init_token = '0'
                    initial_label = to_label(init_token)
                    initial_answers[agent_id] = f"answer: {initial_label}"

                    final_token = str(agent_data.get('answer', init_token))
                    final_label = to_label(final_token)
                    consensus_answers[agent_id] = {
                        'answer': final_label,
                        'answer_changed': (initial_label != final_label)
                    }

                agent_id_keys = set(agent_detailed.keys())
                for agent_id in agent_id_keys:
                    if agent_id not in node_changes:
                        node_changes[agent_id] = []

                    md = consensus_answers.get(agent_id, {})

                    initial_str = initial_answers.get(agent_id, "")
                    initial_val = self._extract_answer_from_string(initial_str)
                    final_val = str(md.get('answer', initial_val))
                    changed = bool(md.get('answer_changed', (initial_val != final_val)))

                    node_changes[agent_id].append({
                        'question': question_id,
                        'initial': initial_val,
                        'final': final_val,
                        'changed': changed
                    })

                    if correct_answer is not None:
                        if str(initial_val).strip().lower() == str(correct_answer).strip().lower():
                            agent_correct[agent_id] = agent_correct.get(agent_id, 0) + 1
                        if str(final_val).strip().lower() == str(correct_answer).strip().lower():
                            agent_final_correct[agent_id] = agent_final_correct.get(agent_id, 0) + 1
                    agent_total[agent_id] = agent_total.get(agent_id, 0) + 1
            else:

                correct_answer = result.get('consensus_evaluation', {}).get('correct_answer')
                initial_answers = result.get('agent_initial_results', {}) or {}
                consensus_answers = result.get('agent_consensus_results', {}) or {}

                if str(config.get('dataset_type', '')).lower() == 'safe':

                    if not initial_answers:
                        ir_list = result.get('individual_responses', []) or []
                        for ir in ir_list:
                            if isinstance(ir, dict) and ir.get('agent_id') is not None:
                                initial_answers[ir['agent_id']] = str(ir.get('answer', ''))
                    if not consensus_answers:
                        aa_map = result.get('agent_answers', {}) or {}
                        for aid, ans in aa_map.items():
                            consensus_answers[aid] = {'answer': str(ans), 'answer_changed': False}

                    norm_initial = {}
                    for aid, ans in initial_answers.items():
                        raw = ans
                        if isinstance(raw, str) and raw.lower().startswith('answer:'):
                            raw = raw.split(':', 1)[1].strip()
                        token = str(raw).strip().lower()
                        lbl = 'safe' if token in ('1', 'safe') else 'unsafe'
                        norm_initial[aid] = f"answer: {lbl}"
                    initial_answers = norm_initial

                    norm_final = {}
                    for aid, ans in consensus_answers.items():
                        if isinstance(ans, dict):
                            token = str(ans.get('answer', '')).strip().lower()
                            changed = bool(ans.get('answer_changed', False))
                        else:
                            token = str(ans).strip().lower()
                            changed = False
                        lbl = 'safe' if token in ('1', 'safe') else 'unsafe'
                        norm_final[aid] = {'answer': lbl, 'answer_changed': changed}
                    consensus_answers = norm_final

                    ce = result.get('consensus_evaluation', {}) or {}
                    ca_raw = ce.get('correct_answer', '')
                    ca_norm = str(ca_raw).strip().lower()
                    correct_answer = 'safe' if ca_norm in ('1','safe') else 'unsafe'

                agent_id_keys = set(initial_answers.keys()) | set(consensus_answers.keys())
                if not agent_id_keys:

                    for resp in result.get('individual_responses', []):
                        if isinstance(resp, dict) and resp.get('agent_id'):
                            agent_id_keys.add(resp['agent_id'])

                for agent_id in agent_id_keys:
                    if agent_id not in node_changes:
                        node_changes[agent_id] = []

                    initial_str = initial_answers.get(agent_id, "")
                    initial_answer = self._extract_answer_from_string(initial_str)

                    consensus_data = consensus_answers.get(agent_id, {})
                    if isinstance(consensus_data, dict):
                        consensus_answer = str(consensus_data.get('answer', initial_answer))
                    else:
                        consensus_answer = str(consensus_data)

                    agent_type = str(config.get('agent_type', '')).lower()
                    if isinstance(consensus_data, dict) and 'answer_changed' in consensus_data:
                        changed = bool(consensus_data['answer_changed'])
                    else:
                        changed = (initial_answer != consensus_answer)
                    node_changes[agent_id].append({
                        'question': question_id,
                        'initial': initial_answer,
                        'final': consensus_answer,
                        'changed': changed
                    })

                    if correct_answer is not None:

                        extracted_answer = initial_answer.strip()
                        expected_answer = str(correct_answer).strip()

                        if extracted_answer.startswith("answer:"):
                            extracted_answer = extracted_answer.replace("answer:", "").strip()

                            if "," in extracted_answer:
                                extracted_answer = extracted_answer.split(",")[0].strip()

                        if extracted_answer == expected_answer:
                            agent_correct[agent_id] += 1

                        final_answer = consensus_answer.strip()

                        if final_answer.startswith("answer:"):
                            final_answer = final_answer.replace("answer:", "").strip()
                            if "," in final_answer:
                                final_answer = final_answer.split(",")[0].strip()

                        if final_answer == expected_answer:
                            agent_final_correct[agent_id] += 1

                    agent_total[agent_id] = agent_total.get(agent_id, 0) + 1

            consensus_evaluation = result.get('consensus_evaluation', {})
            is_consensus_correct = consensus_evaluation.get('is_consensus_correct')
            if is_consensus_correct is None:
                from collections import Counter
                votes = Counter(str(v.get('answer', v) if isinstance(v, dict) else v)
                                for v in consensus_answers.values())
                majority = votes.most_common(1)[0][0] if votes else ''

                def _eq(a: str, b: str) -> bool:
                    aa = str(a).strip().lower(); bb = str(b).strip().lower()
                    def m(x):
                        if x in ('1','safe'): return 'safe'
                        if x in ('0','unsafe'): return 'unsafe'
                        return x
                    return m(aa) == m(bb)
                is_consensus_correct = _eq(majority, correct_answer)
            if is_consensus_correct:
                consensus_correct += 1

        for agent_id in agent_stats:
            if agent_id not in agent_correct:
                agent_correct[agent_id] = 0
            if agent_id not in agent_total:
                agent_total[agent_id] = 0
            if agent_id not in agent_final_correct:
                agent_final_correct[agent_id] = 0
            if agent_id not in node_changes:
                node_changes[agent_id] = []

        try:
            total_answers_sum = sum(int(v or 0) for v in agent_total.values())
            if total_answers_sum == 0 and isinstance(data, dict):
                eval_metrics = data.get('evaluation_metrics', {}) or {}
                node_acc = (eval_metrics.get('node_accuracy', {}) or {}).get('node_accuracies', {}) or {}

                for aid, rec in node_acc.items():

                    if agent_total.get(aid, 0) == 0:
                        agent_total[aid] = int(rec.get('total_count', 0) or 0)
                    if agent_correct.get(aid, 0) == 0:
                        agent_correct[aid] = int(rec.get('correct_count', 0) or 0)
                    if agent_final_correct.get(aid, 0) == 0:
                        agent_final_correct[aid] = int(rec.get('correct_count', 0) or 0)

                def _map_safe_label(val: Any) -> str:
                    s = str(val).strip().lower()
                    if s in ('1', 'safe'):
                        return 'safe'
                    return 'unsafe'
                if all(len(changes) == 0 for changes in node_changes.values()):
                    top_consensus = data.get('consensus_results', []) or []
                    top_questions = data.get('questions', []) or []

                    if top_consensus:
                        cr0 = top_consensus[0]
                        indiv = cr0.get('individual_responses', []) or []

                        if top_questions:
                            qtext = top_questions[0].get('question_text') or str(top_questions[0].get('question', 'unknown'))
                        else:
                            qtext = cr0.get('question_id', 'unknown')
                        for resp in indiv:
                            aid = resp.get('agent_id')
                            if not aid:
                                continue
                            init_lbl = _map_safe_label((resp.get('metadata') or {}).get('initial_answer', resp.get('answer')))
                            final_lbl = _map_safe_label((resp.get('metadata') or {}).get('final_answer', resp.get('answer')))
                            if aid not in node_changes:
                                node_changes[aid] = []
                            node_changes[aid].append({
                                'question': qtext,
                                'initial': init_lbl,
                                'final': final_lbl,
                                'changed': (init_lbl != final_lbl)
                            })
        except Exception:
            pass

        return {
            'format': ('safe' if dataset_type == 'safe' else ('commonsense' if dataset_type == 'commonsense' else 'gsm8k')),
            'node_changes': node_changes,
            'agent_correct': agent_correct,
            'agent_final_correct': agent_final_correct,
            'agent_total': agent_total,
            'consensus_correct': consensus_correct,
            'total_rounds': total_rounds,
            'malicious_agents': malicious_agents,
            'agent_stats': agent_stats,                      
            'config': config
        }

    def _analyze_detailed_gsm8k_format(self, data: List[Dict]) -> Dict:

        node_changes = {}
        malicious_agents = set()
        agent_correct = {}
        agent_total = {}
        consensus_correct = 0
        total_rounds = len(data)

        for item in data:
            question = item.get('question', {})
            question_id = question.get('question', f"question_{item.get('round_index', 1)}")

            if 'Jerry' in question_id:
                question_id = 'gsm8k_jerry'
            elif len(question_id) > 50:
                question_id = f"gsm8k_{item.get('round_index', 1)}"

            agent_detailed = item.get('agent_detailed', {})
            consensus_results = item.get('consensus_results', [])

            answer_changes = {}
            if consensus_results:
                answer_changes = consensus_results[0].get('answer_changes', {})

            for agent_id, details in agent_detailed.items():
                if agent_id not in node_changes:
                    node_changes[agent_id] = []
                    agent_correct[agent_id] = 0
                    agent_total[agent_id] = 0

                if details.get('is_malicious', False):
                    malicious_agents.add(agent_id)

                raw_response = details.get('raw_response', '')
                final_answer = details.get('answer', '')

                original_answer = self._extract_answer_from_string(raw_response)

                answer_changed_from_record = agent_id in answer_changes if answer_changes else False
                if answer_changed_from_record:
                    change_info = answer_changes[agent_id]
                    original_from_changes = change_info.get('original_answer', original_answer)
                    new_answer = change_info.get('new_answer', final_answer)
                else:
                    original_from_changes = original_answer
                    new_answer = final_answer

                actual_changed = str(original_from_changes).strip() != str(new_answer).strip()

                node_changes[agent_id].append({
                    'question_id': question_id,
                    'original_answer': original_from_changes,
                    'final_answer': new_answer,
                    'changed': actual_changed,            
                    'is_malicious': details.get('is_malicious', False)
                })

                agent_total[agent_id] += 1

        return {
            'format': 'detailed_gsm8k',
            'node_changes': node_changes,
            'agent_correct': agent_correct,
            'agent_total': agent_total,
            'consensus_correct': consensus_correct,
            'total_rounds': total_rounds,
            'malicious_agents': list(malicious_agents),
            'config': {}
        }

    def _analyze_safe_format(self, data: List) -> Dict:

        if not isinstance(data, list) or not data:
            return self._analyze_gsm8k_format({'test_results': [], 'agent_stats': {}, 'malicious_agents': []})

        first_item = data[0]
        agent_detailed = first_item.get('agent_detailed', {})
        agent_ids = list(agent_detailed.keys())
        malicious_agents = [aid for aid, info in agent_detailed.items() if info.get('is_malicious', False)]

        topology = 'unknown'
        if 'topology' in first_item:
            topology = first_item['topology']
        elif 'config' in first_item:
            topology = first_item['config'].get('topology', 'unknown')

        config = {
            'topology': topology,
            'agents': len(agent_ids),
            'malicious': len(malicious_agents),
            'rounds': len(data)
        }

        node_changes = {aid: [] for aid in agent_ids}
        agent_correct = {aid: 0 for aid in agent_ids}         
        agent_final_correct = {aid: 0 for aid in agent_ids}         
        agent_total = {aid: 0 for aid in agent_ids}
        consensus_correct = 0

        for idx, item in enumerate(data):
            question_id = f"safe_q{item.get('question', {}).get('id', idx+1)}"

            agent_detailed = item.get('agent_detailed', {})

            correct_answer = '1'

            for agent_id in agent_ids:
                agent_total[agent_id] += 1

                agent_data = agent_detailed.get(agent_id, {})

                raw_response = agent_data.get('raw_response', '')
                if 'answer:' in raw_response:
                    initial_answer = raw_response.split('answer:')[1].split(',')[0].strip()
                else:
                    initial_answer = '0'

                final_answer = str(agent_data.get('answer', initial_answer))

                answer_changed = (initial_answer != final_answer)

                node_changes[agent_id].append({
                    'question': question_id,
                    'initial': initial_answer,
                    'final': final_answer,
                    'changed': answer_changed
                })

                if initial_answer == correct_answer:
                    agent_correct[agent_id] += 1

                if final_answer == correct_answer:
                    agent_final_correct[agent_id] += 1

            consensus_results = item.get('consensus_results', [])
            if consensus_results:
                final_answer = str(consensus_results[-1].get('final_answer', ''))
                is_consensus_correct = (final_answer == correct_answer)
                if is_consensus_correct:
                    consensus_correct += 1

        return {
            'format': 'safe',
            'node_changes': node_changes,
            'agent_correct': agent_correct,
            'agent_final_correct': agent_final_correct,
            'agent_total': agent_total,
            'consensus_correct': consensus_correct,
            'total_rounds': len(data),
            'malicious_agents': malicious_agents,
            'config': config
        }

    def _analyze_safe_byzantine_format(self, data: Dict) -> Dict:

        test_results = data.get('test_results', [])
        config = data.get('test_config', {})
        malicious_agents = config.get('malicious_agents', [])

        node_changes = {}
        agent_correct = {}         
        agent_final_correct = {}         
        agent_total = {}
        consensus_correct = 0
        total_rounds = len(test_results)

        if test_results:
            first_result = test_results[0]
            agent_detailed = first_result.get('agent_detailed', {})
            for agent_id in agent_detailed.keys():
                node_changes[agent_id] = []
                agent_correct[agent_id] = 0
                agent_final_correct[agent_id] = 0
                agent_total[agent_id] = 0

        for result in test_results:
            question_text = result.get('question', {}).get('question', '')

            question_id = question_text[:50] + "..." if len(question_text) > 50 else question_text

            question_data = result.get('question', {})
            question_label = question_data.get('label', 'safe')

            correct_answer = '1'

            consensus_results = result.get('consensus_results', [])
            if consensus_results:
                final_answer = str(consensus_results[-1].get('final_answer', ''))
                is_consensus_correct = (final_answer == correct_answer)
            else:
                is_consensus_correct = False

            if is_consensus_correct:
                consensus_correct += 1

            agent_detailed = result.get('agent_detailed', {})

            for agent_id in node_changes.keys():
                agent_total[agent_id] += 1

                agent_data = agent_detailed.get(agent_id, {})

                raw_response = agent_data.get('raw_response', '')
                if 'answer:' in raw_response:
                    initial_answer = raw_response.split('answer:')[1].split(',')[0].strip()
                else:
                    initial_answer = '0'

                final_answer = str(agent_data.get('answer', initial_answer))

                answer_changed = (initial_answer != final_answer)

                node_changes[agent_id].append({
                    'question': question_id,
                    'initial': initial_answer,
                    'final': final_answer,
                    'changed': answer_changed
                })

                if initial_answer == correct_answer:
                    agent_correct[agent_id] += 1

                if final_answer == correct_answer:
                    agent_final_correct[agent_id] += 1

        return {
            'format': 'safe_byzantine',
            'node_changes': node_changes,
            'agent_correct': agent_correct,
            'agent_final_correct': agent_final_correct,
            'agent_total': agent_total,
            'consensus_correct': consensus_correct,
            'total_rounds': total_rounds,
            'malicious_agents': malicious_agents,
            'config': config
        }

    def _analyze_prompt_probe_format(self, data: Dict) -> Dict:

        question_results = data.get('question_results', [])
        config = data.get('test_config', {})
        malicious_agents = data.get('malicious_agents', [])

        node_changes = {}
        agent_correct = {}              
        agent_final_correct = {}              
        agent_total = {}
        consensus_correct = 0
        total_rounds = len(question_results)

        if question_results:
            first_result = question_results[0]
            initial_responses = first_result.get('initial_responses', {})
            for agent_id in initial_responses.keys():
                node_changes[agent_id] = []
                agent_correct[agent_id] = 0
                agent_final_correct[agent_id] = 0
                agent_total[agent_id] = 0

        dataset_type = str(config.get('dataset_type', '')).lower()
        is_safe_dataset = dataset_type == 'safe'

        for result in question_results:
            question_data = result.get('question', {})
            question_text = question_data.get('question', '')

            question_id = question_text[:50] + "..." if len(question_text) > 50 else question_text

            correct_answer = question_data.get('answer', '')
            consensus_rounds = result.get('consensus_rounds', [])
            is_consensus_correct = False

            if consensus_rounds:
                final_consensus = consensus_rounds[-1]

                consensus_result = final_consensus.get('consensus_result', {})
                consensus_answer = consensus_result.get('answer', '')

                if is_safe_dataset:

                    if consensus_answer in ['0', '1']:
                        consensus_answer_mapped = 'safe' if consensus_answer == '1' else 'unsafe'
                    else:
                        consensus_answer_mapped = consensus_answer

                    if correct_answer in ['0', '1']:
                        correct_answer_mapped = 'safe' if correct_answer == '1' else 'unsafe'
                    else:
                        correct_answer_mapped = correct_answer

                    is_consensus_correct = (consensus_answer_mapped == correct_answer_mapped)
                else:
                    is_consensus_correct = (consensus_answer == correct_answer)

            if is_consensus_correct:
                consensus_correct += 1

            initial_responses = result.get('initial_responses', {})
            consensus_rounds = result.get('consensus_rounds', [])

            final_answers = {}
            if consensus_rounds:
                final_round = consensus_rounds[-1]
                agent_responses = final_round.get('agent_responses', {})
                for agent_id, response in agent_responses.items():
                    if isinstance(response, dict):
                        final_answers[agent_id] = response.get('final_answer', response.get('answer', ''))
                    else:
                        final_answers[agent_id] = str(response)

            for agent_id in initial_responses.keys():

                if agent_id not in node_changes:
                    node_changes[agent_id] = []
                    agent_correct[agent_id] = 0
                    agent_total[agent_id] = 0

                agent_total[agent_id] += 1

                initial_response = initial_responses.get(agent_id, {})
                if isinstance(initial_response, dict):
                    initial_answer = initial_response.get('answer', '')
                else:
                    initial_answer = str(initial_response)

                final_answer = final_answers.get(agent_id, initial_answer)

                answer_changed = initial_answer != final_answer

                node_changes[agent_id].append({
                    'question': question_id,
                    'initial': initial_answer,
                    'final': final_answer,
                    'changed': answer_changed
                })

                if is_safe_dataset:

                    initial_answer_mapped = 'safe' if initial_answer == '1' else 'unsafe' if initial_answer == '0' else initial_answer
                    correct_answer_for_compare = 'safe' if correct_answer == '1' else 'unsafe' if correct_answer == '0' else correct_answer
                    if initial_answer_mapped == correct_answer_for_compare:
                        agent_correct[agent_id] += 1
                else:

                    if initial_answer == correct_answer:
                        agent_correct[agent_id] += 1

                if is_safe_dataset:

                    final_answer_mapped = 'safe' if final_answer == '1' else 'unsafe' if final_answer == '0' else final_answer
                    if final_answer_mapped == correct_answer_for_compare:
                        agent_final_correct[agent_id] += 1
                else:

                    if final_answer == correct_answer:
                        agent_final_correct[agent_id] += 1

        return {
            'format': 'prompt_probe',
            'node_changes': node_changes,
            'agent_correct': agent_correct,
            'agent_final_correct': agent_final_correct,
            'agent_total': agent_total,
            'consensus_correct': consensus_correct,
            'total_rounds': total_rounds,
            'malicious_agents': malicious_agents,
            'config': config
        }

    def _calculate_statistics(self, analysis_data: Dict) -> Dict:

        node_changes = analysis_data['node_changes']
        malicious_agents = analysis_data['malicious_agents']

        change_stats = {}
        normal_changes = []
        malicious_changes = []

        for agent_id, changes in node_changes.items():
            total_questions = len(changes)
            changed_count = sum(1 for change in changes if change['changed'])
            change_rate = changed_count / total_questions if total_questions > 0 else 0

            change_stats[agent_id] = {
                'changed_count': changed_count,
                'total_questions': total_questions,
                'change_rate': change_rate,
                'is_malicious': agent_id in malicious_agents
            }

            if agent_id in malicious_agents:
                malicious_changes.append(change_rate)
            else:
                normal_changes.append(change_rate)

        avg_normal_change = sum(normal_changes) / len(normal_changes) if normal_changes else 0
        avg_malicious_change = sum(malicious_changes) / len(malicious_changes) if malicious_changes else 0

        agent_correct = analysis_data.get('agent_correct', {})
        agent_final_correct = analysis_data.get('agent_final_correct', {})
        agent_total = analysis_data.get('agent_total', {})

        accuracy_stats = {}
        normal_initial_accuracies = []
        normal_final_accuracies = []
        malicious_initial_accuracies = []
        malicious_final_accuracies = []

        total_initial_correct = 0
        total_final_correct = 0
        total_answers = 0

        for agent_id in agent_correct.keys():
            initial_correct = agent_correct.get(agent_id, 0)
            final_correct = agent_final_correct.get(agent_id, initial_correct)                                       
            total = agent_total.get(agent_id, 0)

            initial_accuracy = initial_correct / total if total > 0 else 0
            final_accuracy = final_correct / total if total > 0 else 0

            accuracy_stats[agent_id] = {
                'initial_correct': initial_correct,
                'final_correct': final_correct,
                'total': total,
                'initial_accuracy': initial_accuracy,
                'final_accuracy': final_accuracy,
                'is_malicious': agent_id in malicious_agents
            }

            total_initial_correct += initial_correct
            total_final_correct += final_correct
            total_answers += total

            if agent_id in malicious_agents:
                malicious_initial_accuracies.append(initial_accuracy)
                malicious_final_accuracies.append(final_accuracy)
            else:
                normal_initial_accuracies.append(initial_accuracy)
                normal_final_accuracies.append(final_accuracy)

        avg_normal_initial = sum(normal_initial_accuracies) / len(normal_initial_accuracies) if normal_initial_accuracies else 0
        avg_normal_final = sum(normal_final_accuracies) / len(normal_final_accuracies) if normal_final_accuracies else 0
        avg_malicious_initial = sum(malicious_initial_accuracies) / len(malicious_initial_accuracies) if malicious_initial_accuracies else 0
        avg_malicious_final = sum(malicious_final_accuracies) / len(malicious_final_accuracies) if malicious_final_accuracies else 0

        overall_initial_accuracy = total_initial_correct / total_answers if total_answers > 0 else 0
        overall_final_accuracy = total_final_correct / total_answers if total_answers > 0 else 0
        improvement = overall_final_accuracy - overall_initial_accuracy

        return {
            'change_stats': change_stats,
            'avg_normal_change': avg_normal_change,
            'avg_malicious_change': avg_malicious_change,
            'accuracy_stats': accuracy_stats,
            'avg_normal_initial_accuracy': avg_normal_initial,
            'avg_normal_final_accuracy': avg_normal_final,
            'avg_malicious_initial_accuracy': avg_malicious_initial,
            'avg_malicious_final_accuracy': avg_malicious_final,
            'overall_metrics': {
                'initial_accuracy': overall_initial_accuracy,
                'final_accuracy': overall_final_accuracy,
                'improvement': improvement,
                'total_initial_correct': total_initial_correct,
                'total_final_correct': total_final_correct,
                'total_answers': total_answers
            }
        }

    def analyze_test_result(self, test_result: Dict, result_dir: str) -> str:

        try:

            try:
                cfg = (test_result.get('test_config') or {})

                def _infer_method_name(payload_cfg: dict, payload: dict) -> str:
                    mt = (payload.get('method_type')
                          or (payload_cfg.get('method_type') if isinstance(payload_cfg, dict) else None))
                    s = str(mt).lower() if mt else ''
                    if s == 'gsm8k':
                        return 'pilot'
                    if s == 'prompt_probe':
                        return 'prompt'
                    if s in ('decoder_probe', 'decoder'):
                        return 'decoder'
                    return 'pilot'
                method_name = _infer_method_name(cfg, test_result)

                if not cfg:
                    cfg = {}

                if 'dataset_type' not in cfg and 'metadata' in test_result:
                    meta = test_result.get('metadata', {})
                    if isinstance(meta, dict) and 'config' in meta:
                        meta_cfg = meta['config']
                        if isinstance(meta_cfg, dict):
                            cfg = dict(cfg)
                            cfg.update(meta_cfg)

                from ..utils.naming import infer_from_config_or_result
                _dir, prefix = infer_from_config_or_result(cfg, test_result, method_name=method_name)
            except Exception:
                from datetime import datetime as _dt
                prefix = f"analysis_{_dt.now().strftime('%Y%m%d_%H%M%S')}"
            output_path = os.path.join(result_dir, f"{prefix}_report.txt")

            analysis_data = self._analyze_legacy_format(test_result)

            stats = self._calculate_statistics(analysis_data)

            self._generate_detailed_report(analysis_data, stats, output_path, test_result)

            try:
                default_core = os.path.join(result_dir, 'core_metrics.png')
                if os.path.exists(default_core):
                    os.replace(default_core, os.path.join(result_dir, f"{prefix}_core_metrics.png"))
                default_attack = os.path.join(result_dir, 'attack_effect_analysis.png')
                if os.path.exists(default_attack):
                    os.replace(default_attack, os.path.join(result_dir, f"{prefix}_attack_effect.png"))
            except Exception:
                pass

            try:

                orig_savefig = None
                self._generate_topology_structure(test_result, result_dir)

                default_topo = os.path.join(result_dir, 'topology_structure.png')
                if os.path.exists(default_topo):
                    os.replace(default_topo, os.path.join(result_dir, f"{prefix}_topology.png"))
            except Exception:
                pass

            return output_path

        except Exception as e:
            print(f"分析测试结果失败: {e}")
            return None

    def analyze_test_result_with_custom_path(self, test_result: Dict, result_dir: str, custom_output_path: str) -> str:

        try:

            analysis_data = self._analyze_legacy_format(test_result)

            stats = self._calculate_statistics(analysis_data)

            self._generate_detailed_report(analysis_data, stats, custom_output_path, test_result)

            self._generate_topology_structure(test_result, result_dir)

            return custom_output_path

        except Exception as e:
            print(f"分析测试结果失败: {e}")
            return None

    def analyze(self, json_path: str, output_path: str):

        with open(json_path, 'r', encoding='utf-8') as f:
            test_result = json.load(f)

        result_dir = os.path.dirname(output_path)

        return self.analyze_test_result_with_custom_path(test_result, result_dir, output_path)

    def analyze_in_memory(self, test_result: Dict, output_path: str) -> str:

        result_dir = os.path.dirname(output_path)
        os.makedirs(result_dir, exist_ok=True)
        return self.analyze_test_result_with_custom_path(test_result, result_dir, output_path)

    def analyze_and_save_unified(self, test_result: Dict, output_dir: str, json_filename: str = None) -> str:

        try:

            os.makedirs(output_dir, exist_ok=True)

            if json_filename is None:

                topology_type = test_result.get('topology_type', 'chain')
                agent_count = test_result.get('agent_count', 5)
                malicious_count = test_result.get('malicious_count', 1)

                json_filename = f"traditional_{topology_type}_{agent_count}_{malicious_count}.json"

            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(test_result, f, ensure_ascii=False, indent=2, default=str)
            print(f"JSON文件已生成: {json_filename}")

            analysis_path = os.path.join(output_dir, "comprehensive_analysis.txt")
            result_path = self.analyze_test_result_with_custom_path(test_result, output_dir, analysis_path)

            print(f"统一输出完成，所有文件在: {output_dir}")
            return result_path

        except Exception as e:
            print(f"统一输出失败: {e}")
            return None

    def _analyze_legacy_format(self, data: Dict) -> Dict:

        if isinstance(data, list):

            return self._analyze_safe_format(data)
        elif 'question_results' in data and 'malicious_agents' in data:

            return self._analyze_prompt_probe_format(data)
        elif 'test_results' in data:

            config = data.get('test_config', {})
            agent_type = config.get('agent_type', '')
            dataset_type = config.get('dataset_type', '')

            if agent_type == 'safe':

                return self._analyze_gsm8k_format(data)
            else:

                return self._analyze_gsm8k_format(data)
        elif 'questions' in data and 'consensus_results' in data:

            questions = data.get('questions', [])
            consensus_results = data.get('consensus_results', [])
            converted_data = dict(data)         
            converted_data['test_results'] = []

            try:
                base_cfg = (data.get('metadata') or {}).get('config') or {}
                existing_cfg = converted_data.get('test_config') or {}
                merged_cfg = dict(existing_cfg)
                for key in ['dataset_type', 'agent_type', 'topology_type', 'topology',
                            'num_agents', 'agents', 'num_malicious', 'malicious']:
                    if key in base_cfg and key not in merged_cfg:
                        merged_cfg[key] = base_cfg[key]
                converted_data['test_config'] = merged_cfg
            except Exception:
                pass

            agent_stats = {}
            if consensus_results and 'individual_responses' in consensus_results[0]:
                for response in consensus_results[0]['individual_responses']:
                    agent_id = response.get('agent_id')
                    if agent_id:
                        agent_metadata = response.get('metadata', {})
                        agent_stats[agent_id] = {
                            'agent_type': agent_metadata.get('agent_type', 'unknown'),
                            'is_malicious': agent_metadata.get('is_malicious', False),
                            'model_info': agent_metadata.get('dataset_type', 'unknown')
                        }
            converted_data['agent_stats'] = agent_stats

            if 'metadata' in data and 'malicious_agents' in data['metadata']:
                converted_data['malicious_agents'] = data['metadata']['malicious_agents']
            elif 'malicious_agents' not in converted_data:

                malicious_agents = [agent_id for agent_id, stats in agent_stats.items() 
                                  if stats.get('is_malicious', False)]
                converted_data['malicious_agents'] = malicious_agents

            converted_data['round_results'] = []                     

            def _safe_norm(s: str) -> str:
                t = str(s).strip().lower()
                if t in ('1', 'safe'):
                    return 'safe'
                if t in ('0', 'unsafe'):
                    return 'unsafe'
                return t

            for idx, (question_data, consensus_data) in enumerate(zip(questions, consensus_results)):

                expected_answer = question_data.get('correct_answer', question_data.get('answer', ''))
                final_consensus_answer = consensus_data.get('consensus_answer', '')

                if str(converted_data.get('test_config', {}).get('dataset_type', '')).lower() == 'safe':
                    expected_norm = _safe_norm(expected_answer)
                    final_norm = _safe_norm(final_consensus_answer)
                    is_correct_flag = (expected_norm == final_norm)
                else:
                    is_correct_flag = (str(expected_answer).strip() == str(final_consensus_answer).strip())

                round_result = {
                    'round_number': idx + 1,
                    'question': {
                        'id': question_data.get('question_id', f'question_{idx+1}'),
                        'question': question_data.get('question_text', ''),
                        'answer': question_data.get('correct_answer', ''),
                        '_dataset_type': question_data.get('question_type', 'math')
                    },
                    'agent_answers': {},            
                    'consensus_results': {},                 
                    'agent_results': [],                     
                    'individual_responses': [],        
                    'is_correct': bool(is_correct_flag)
                }

                if 'individual_responses' in consensus_data:
                    for response in consensus_data['individual_responses']:
                        agent_id = response.get('agent_id')
                        md = response.get('metadata', {}) or {}
                        initial_answer = (
                            md.get('before_degree1_answer',
                                   md.get('initial_answer', response.get('answer', '')))
                        )

                        final_answer = md.get('final_answer', response.get('answer', ''))

                        if agent_id:

                            round_result['agent_answers'][agent_id] = initial_answer

                            round_result['consensus_results'][agent_id] = final_answer

                            round_result['individual_responses'].append({
                                'agent_id': agent_id,
                                'answer': final_answer,          
                                'metadata': response.get('metadata', {})
                            })

                            round_result['agent_results'].append({
                                'agent_id': agent_id,
                                'answer': final_answer,          
                                'initial_answer': initial_answer,
                                'is_correct': (str(final_answer).strip().lower() == str(expected_answer).strip().lower()),
                                'initial_is_correct': (str(initial_answer).strip().lower() == str(expected_answer).strip().lower())
                            })

                converted_data['round_results'].append(round_result)

            converted_data['test_results'] = []
            for round_data in converted_data['round_results']:
                test_result = {
                    'round_number': round_data['round_number'],
                    'question': round_data['question'],
                    'agent_initial_results': {},
                    'agent_consensus_results': {},
                    'individual_responses': round_data['individual_responses'],
                    'consensus_evaluation': {
                        'is_consensus_correct': round_data['is_correct'],
                        'correct_answer': round_data['question']['answer'],
                        'final_consensus_answer': str(consensus_results.get('final_answer', final_consensus_answer)) if isinstance(consensus_results, dict) else final_consensus_answer
                    }
                }

                for agent_id, init_answer in round_data['agent_answers'].items():
                    formatted_answer = f"answer: {init_answer}"

                    test_result['agent_initial_results'][agent_id] = formatted_answer

                    final_ans = str(round_data['consensus_results'].get(agent_id, final_consensus_answer))
                    test_result['agent_consensus_results'][agent_id] = {
                        'answer': final_ans,
                        'answer_changed': (str(final_ans).strip().lower() != str(init_answer).strip().lower())
                    }

                converted_data['test_results'].append(test_result)

            if 'questions' in converted_data:
                del converted_data['questions']
            if 'consensus_results' in converted_data:
                del converted_data['consensus_results']

            analysis = self._analyze_gsm8k_format(converted_data)
            if isinstance(analysis, dict):
                analysis['round_results'] = converted_data.get('round_results', [])
                analysis['test_config'] = converted_data.get('test_config', {})
                analysis['malicious_agents'] = converted_data.get('malicious_agents', [])
            return analysis
        elif 'round_results' not in data:

            return self._analyze_gsm8k_format(data)

        round_results = data.get('round_results', [])
        malicious_agents = data.get('malicious_agents', [])
        config = data.get('test_config', {})

        agent_stats = {}
        if round_results:
            first_round = round_results[0]
            agent_details = first_round.get('agent_details', {})
            for agent_id, details in agent_details.items():
                agent_stats[agent_id] = {
                    'agent_type': details.get('agent_type', 'unknown'),
                    'is_malicious': details.get('is_malicious', False),
                    'model_info': str(details.get('model_info', 'unknown'))
                }

        converted_data = {
            'test_results': [],
            'agent_stats': agent_stats,
            'malicious_agents': malicious_agents,
            'test_config': config
        }

        for idx, result in enumerate(round_results):

            question = result.get('question', {})
            correct_answer = question.get('answer', '')
            dataset_type = question.get('_dataset_type', '')

            is_safe_dataset = (dataset_type == 'safe' or 
                             correct_answer in ['safe', 'unsafe'] or
                             'safe' in str(question.get('id', '')).lower())

            converted_result = {
                'round_number': idx + 1,
                'question': question,
                'agent_initial_results': {},
                'agent_consensus_results': {},
                'consensus_evaluation': {
                    'is_consensus_correct': result.get('is_correct', False),
                    'correct_answer': correct_answer            
                }
            }

            agent_answers = result.get('agent_answers', {})
            for agent_id, answer in agent_answers.items():
                if is_safe_dataset:

                    mapped_answer = 'safe' if str(answer) == '1' else 'unsafe'
                    converted_result['agent_initial_results'][agent_id] = f"answer: {mapped_answer}"
                else:

                    converted_result['agent_initial_results'][agent_id] = f"answer: {answer}"

            consensus_results = result.get('consensus_results', {})
            for agent_id, answer in consensus_results.items():
                if is_safe_dataset:

                    mapped_answer = 'safe' if str(answer) == '1' else 'unsafe'
                    converted_result['agent_consensus_results'][agent_id] = {'answer': mapped_answer}
                else:

                    converted_result['agent_consensus_results'][agent_id] = {'answer': answer}

            converted_data['test_results'].append(converted_result)

        return self._analyze_gsm8k_format(converted_data)

    def _generate_detailed_report(self, analysis_data: Dict, stats: Dict, output_path: str, original_data: Dict = None):

        config = analysis_data['config']
        malicious_agents = analysis_data['malicious_agents']
        node_changes = analysis_data['node_changes']
        change_stats = stats['change_stats']

        with open(output_path, 'w', encoding='utf-8') as f:

            f.write("="*80 + "\n")
            f.write("拜占庭容错系统 - 综合分析报告\n")
            f.write("="*80 + "\n")
            f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

            data_format = analysis_data.get('format', 'unknown')
            if data_format in ['safe', 'safe_byzantine']:
                config_type = 'Safe 拜占庭容错系统'
            elif data_format == 'gsm8k':
                config_type = 'GSM8K 拜占庭容错系统'
            elif data_format == 'commonsense':
                config_type = 'CommonsenseQA 拜占庭容错系统'
            else:
                config_type = '拜占庭容错系统'
            f.write(f"测试配置: {config_type}\n")
            f.write(f"网络拓扑: {config.get('topology_type', config.get('topology', '未知'))}\n")

            num_agents = config.get('num_agents', config.get('agents', len(analysis_data.get('node_changes', {}))))
            num_malicious = config.get('num_malicious', config.get('malicious', len(malicious_agents)))
            f.write(f"节点总数: {num_agents}\n")
            f.write(f"恶意节点数量: {num_malicious}\n")
            f.write(f"恶意节点列表: {malicious_agents}\n\n")

            f.write("1. 核心指标分析\n")
            f.write("-"*40 + "\n")

            f.write("1.1 节点准确率分析\n")
            total_rounds_val = analysis_data.get('total_rounds')
            try:
                if not total_rounds_val and isinstance(analysis_data, dict):
                    rr = analysis_data.get('round_results') or []
                    total_rounds_val = len(rr)
            except Exception:
                pass
            f.write(f"  总问题数: {total_rounds_val}\n")

            normal_initial_accuracies = []
            normal_final_accuracies = []
            malicious_initial_accuracies = []
            malicious_final_accuracies = []

            agent_initial_correct = {}
            agent_final_correct = {}
            agent_total = {}

            if original_data and 'consensus_results' in original_data:

                questions = original_data.get('questions', [])
                correct_answers = {q.get('question_id'): q.get('correct_answer') for q in questions}

                def _map_commonsense_letter_to_text(answer: str, question_id: str) -> str:

                    if not answer or len(answer) != 1 or answer.upper() not in 'ABCDE':
                        return answer

                    for q in original_data.get('questions', []):
                        if q.get('question_id') == question_id:
                            choices = q.get('metadata', {}).get('choices', [])
                            if choices:
                                idx = ord(answer.upper()) - 65                 
                                if 0 <= idx < len(choices):
                                    return choices[idx]
                    return answer

                for consensus_round in original_data['consensus_results']:
                    question_id = consensus_round.get('question_id')
                    correct_answer = correct_answers.get(question_id)

                    for resp in consensus_round.get('individual_responses', []):
                        agent_id = resp.get('agent_id')
                        if not agent_id:
                            continue

                        if agent_id not in agent_initial_correct:
                            agent_initial_correct[agent_id] = 0
                            agent_final_correct[agent_id] = 0
                            agent_total[agent_id] = 0

                        agent_total[agent_id] += 1

                        metadata = resp.get('metadata', {})
                        initial_answer = str(metadata.get('initial_answer', resp.get('answer', ''))).strip()
                        final_answer = str(metadata.get('final_answer', resp.get('answer', ''))).strip()

                        if str(correct_answer).lower() in ['safe', 'unsafe']:
                            initial_answer = 'safe' if initial_answer == '1' else 'unsafe' if initial_answer == '0' else initial_answer
                            final_answer = 'safe' if final_answer == '1' else 'unsafe' if final_answer == '0' else final_answer
                        else:

                            initial_answer = _map_commonsense_letter_to_text(initial_answer, question_id)
                            final_answer = _map_commonsense_letter_to_text(final_answer, question_id)

                        if initial_answer.lower() == str(correct_answer).lower():
                            agent_initial_correct[agent_id] += 1
                        if final_answer.lower() == str(correct_answer).lower():
                            agent_final_correct[agent_id] += 1

                for agent_id in agent_total.keys():
                    total = agent_total[agent_id]
                    if total == 0:
                        continue
                    initial_acc = agent_initial_correct[agent_id] / total
                    final_acc = agent_final_correct[agent_id] / total

                    if agent_id in malicious_agents:
                        malicious_initial_accuracies.append(initial_acc)
                        malicious_final_accuracies.append(final_acc)
                    else:
                        normal_initial_accuracies.append(initial_acc)
                        normal_final_accuracies.append(final_acc)

            else:

                for agent_id, correct in analysis_data.get('agent_correct', {}).items():
                    total = analysis_data.get('agent_total', {}).get(agent_id, 0)
                    initial_accuracy = correct / total if total > 0 else 0
                    final_accuracy = analysis_data.get('agent_final_correct', {}).get(agent_id, 0) / total if total > 0 else 0

                    if agent_id in malicious_agents:
                        malicious_initial_accuracies.append(initial_accuracy)
                        malicious_final_accuracies.append(final_accuracy)
                    else:
                        normal_initial_accuracies.append(initial_accuracy)
                        normal_final_accuracies.append(final_accuracy)

            if normal_initial_accuracies:
                f.write(f"  正常节点 ({len(normal_initial_accuracies)}个):\n")
                f.write(f"    初始平均准确率: {sum(normal_initial_accuracies)/len(normal_initial_accuracies):.2%}\n")
                f.write(f"    最终平均准确率: {sum(normal_final_accuracies)/len(normal_final_accuracies):.2%}\n")
                f.write(f"    初始准确率范围: {min(normal_initial_accuracies):.2%} - {max(normal_initial_accuracies):.2%}\n")
                f.write(f"    最终准确率范围: {min(normal_final_accuracies):.2%} - {max(normal_final_accuracies):.2%}\n")

            if malicious_initial_accuracies:
                f.write(f"  恶意节点 ({len(malicious_initial_accuracies)}个):\n")
                f.write(f"    初始平均准确率: {sum(malicious_initial_accuracies)/len(malicious_initial_accuracies):.2%}\n")
                f.write(f"    最终平均准确率: {sum(malicious_final_accuracies)/len(malicious_final_accuracies):.2%}\n")
                f.write(f"    初始准确率范围: {min(malicious_initial_accuracies):.2%} - {max(malicious_initial_accuracies):.2%}\n")
                f.write(f"    最终准确率范围: {min(malicious_final_accuracies):.2%} - {max(malicious_final_accuracies):.2%}\n")

            agent_type_flag = str(config.get('agent_type', '')).lower()
            total_initial_correct = 0
            total_final_correct = 0
            total_answers = 0

            if original_data and isinstance(original_data, dict) and original_data.get('consensus_results') and agent_total:
                total_initial_correct = sum(int(v or 0) for v in agent_initial_correct.values())
                total_final_correct = sum(int(v or 0) for v in agent_final_correct.values())
                total_answers = sum(int(v or 0) for v in agent_total.values())
            else:

                agent_correct_vals = analysis_data.get('agent_correct', {}) or {}
                agent_final_correct_vals = analysis_data.get('agent_final_correct', {}) or {}
                agent_total_vals = analysis_data.get('agent_total', {}) or {}
                total_initial_correct = sum(int(v or 0) for v in agent_correct_vals.values())
                total_final_correct = sum(int(v or 0) for v in agent_final_correct_vals.values())
                total_answers = sum(int(v or 0) for v in agent_total_vals.values())

            initial_accuracy = (total_initial_correct / total_answers) if total_answers > 0 else 0.0
            final_accuracy = (total_final_correct / total_answers) if total_answers > 0 else 0.0
            improvement = final_accuracy - initial_accuracy

            f.write("\n1.2 总体节点准确率分析\n")

            f.write(f"  总节点初始准确率: {initial_accuracy:.2%} ({total_initial_correct}/{total_answers})\n")
            f.write(f"  总节点最终准确率: {final_accuracy:.2%} ({total_final_correct}/{total_answers})\n")
            f.write(f"  拜占庭容错改进: {improvement:+.2%}\n")

            f.write("\n2. 节点答案变化分析\n")
            f.write("-"*40 + "\n")
            f.write(f"总节点数: {len(node_changes)}\n")
            f.write(f"恶意节点: {malicious_agents}\n")
            f.write("计算说明: 答案变化率 = 改变答案的问题数 / 总问题数\n")
            f.write(f"正常节点平均答案变化率: {stats['avg_normal_change']:.2%}\n")
            f.write(f"恶意节点平均答案变化率: {stats['avg_malicious_change']:.2%}\n\n")

            f.write("各节点详细变化情况:\n")

            question_text_map = {}
            if original_data and 'questions' in original_data:
                for q in original_data.get('questions', []):
                    q_id = q.get('question_id', '')
                    q_text = q.get('question_text', q.get('question', ''))
                    if q_id and q_text:
                        question_text_map[q_id] = q_text

            for agent_id, changes in node_changes.items():
                stat = change_stats[agent_id]
                role = "恶意" if stat['is_malicious'] else "正常"
                f.write(f"  {agent_id} ({role}): {stat['changed_count']}/{stat['total_questions']} 变化 ({stat['change_rate']:.2%})\n")

                for change in changes:

                    question_key = 'question' if 'question' in change else 'question_id'
                    initial_key = 'initial' if 'initial' in change else 'original_answer'
                    final_key = 'final' if 'final' in change else 'final_answer'

                    question_id = change.get(question_key, 'unknown')

                    if isinstance(question_id, str) and question_id in question_text_map:
                        question_display = question_text_map[question_id]
                        if len(question_display) > 60:
                            question_display = question_display[:60] + "..."
                    elif isinstance(question_id, str) and question_id.startswith('gsm8k_'):
                        q_list = (original_data or {}).get('questions', []) if isinstance(original_data, dict) else []
                        try:
                            idx = int(question_id.split('_')[-1]) - 1
                            if 0 <= idx < len(q_list):
                                question_display = str(q_list[idx].get('question_text') or q_list[idx].get('question') or question_id)
                            else:
                                question_display = question_id
                        except Exception:
                            question_display = question_id
                    else:

                        q_str = str(question_id)
                        question_display = q_str[:60] + "..." if len(q_str) > 60 else q_str

                    initial_value = change.get(initial_key, '')
                    final_value = change.get(final_key, '')

                    def format_value(value):
                        if isinstance(value, str) and len(value) > 50:

                            return value[:50] + "..."
                        return str(value)

                    if isinstance(initial_value, (int, float)) and isinstance(final_value, (int, float)):

                        initial_display = str(initial_value)
                        final_display = str(final_value)
                    else:

                        initial_display = format_value(initial_value)
                        final_display = format_value(final_value)

                    if change['changed']:
                        f.write(f"    {question_display}: {initial_display} -> {final_display} (变化)\n")
                    else:
                        f.write(f"    {question_display}: {final_display} (不变)\n")

            f.write("\n3. 攻击效果分析\n")
            f.write("-"*40 + "\n")
            f.write(f"网络拓扑: {config.get('topology_type', config.get('topology', '未知'))}\n")
            f.write(f"节点配置: {num_agents}个节点，{num_malicious}个恶意节点\n")
            f.write(f"恶意节点列表: {malicious_agents}\n")

            malicious_ratio = num_malicious / num_agents if num_agents > 0 else 0
            f.write(f"恶意节点比例: {malicious_ratio:.2%}\n")

            total_rounds = 0
            consensus_correct_val = 0
            try:

                consensus_results = []
                if isinstance(original_data, dict) and original_data.get('consensus_results'):
                    consensus_results = original_data.get('consensus_results', [])
                elif isinstance(analysis_data, dict) and analysis_data.get('consensus_results'):
                    consensus_results = analysis_data.get('consensus_results', [])

                if not consensus_results:
                    source_rounds = []
                    if isinstance(original_data, dict) and original_data.get('round_results'):
                        source_rounds = original_data.get('round_results', [])
                    elif isinstance(analysis_data, dict) and analysis_data.get('round_results'):
                        source_rounds = analysis_data.get('round_results', [])

                    for rd in source_rounds:
                        total_rounds += 1
                        exp = str(rd.get('question', {}).get('answer', '')).strip().lower()
                        fc = rd.get('final_consensus') or rd.get('consensus_result') or rd.get('consensus_answer')
                        if fc is None:
                            cr = rd.get('consensus_results', {}) or {}
                            if cr:
                                from collections import Counter
                                cnt = Counter(str(v).strip().lower() for v in cr.values())
                                if cnt:
                                    fc = max(cnt.keys(), key=lambda k: cnt[k])
                        if fc is not None and exp:
                            dataset_type = str((analysis_data.get('config') or {}).get('dataset_type', '')).lower()
                            if dataset_type == 'safe':
                                def _ns(x: str) -> str:
                                    x = str(x).strip().lower()
                                    return x if x in ('safe','unsafe') else ('safe' if x=='1' else 'unsafe')
                                if _ns(fc) == _ns(exp):
                                    consensus_correct_val += 1
                            else:
                                if str(fc).strip().lower() == exp:
                                    consensus_correct_val += 1
                else:

                    for cr in consensus_results:
                        total_rounds += 1

                        dataset_type = str((analysis_data.get('config') or {}).get('dataset_type', '')).lower()
                        if dataset_type == 'safe':
                            def _ns(x: str) -> str:
                                x = str(x).strip().lower()
                                return 'safe' if x in ('1', 'safe') else 'unsafe'
                            consensus_ans = _ns(cr.get('consensus_answer', ''))
                            qid = cr.get('question_id')
                            questions = original_data.get('questions', []) if isinstance(original_data, dict) else []
                            correct_ans = None
                            for q in questions:
                                if q.get('question_id') == qid:
                                    correct_ans = _ns(q.get('correct_answer', ''))
                                    break
                            if correct_ans and consensus_ans == correct_ans:
                                consensus_correct_val += 1
                        else:
                            if 'is_correct' in cr and cr['is_correct'] is not None:
                                if cr['is_correct']:
                                    consensus_correct_val += 1
                            else:

                                consensus_ans = str(cr.get('consensus_answer', '')).strip().lower()

                                qid = cr.get('question_id')
                                questions = original_data.get('questions', []) if isinstance(original_data, dict) else []
                                correct_ans = None
                                for q in questions:
                                    if q.get('question_id') == qid:
                                        correct_ans = str(q.get('correct_answer', '')).strip().lower()
                                        break
                                if correct_ans and consensus_ans == correct_ans:
                                    consensus_correct_val += 1
            except Exception as e:

                import traceback
                print(f"Warning: Error calculating fault tolerance: {e}")
                traceback.print_exc()
            fault_tolerance_rate = (consensus_correct_val/total_rounds) if total_rounds>0 else 0.0
            attack_success_rate = max(0.0, min(1.0, 1.0 - fault_tolerance_rate))

            attack_denominator = analysis_data.get('total_rounds', total_rounds)
            attack_success = int(round(attack_success_rate * attack_denominator + 1e-9))

            f.write(f"攻击成功次数: {attack_success}/{analysis_data['total_rounds']}\n")
            f.write(f"攻击成功率: {attack_success_rate:.2%}\n")
            f.write(f"系统容错能力: {fault_tolerance_rate:.2%}\n")

            f.write("\n" + "="*80 + "\n")
            f.write("报告结束\n")
            f.write("="*80 + "\n")

    def _generate_topology_structure(self, test_result: Dict, result_dir: str):

        try:
            import matplotlib.pyplot as plt
            import importlib, importlib.util

            nx = None
            try:
                if importlib.util.find_spec("networkx") is not None:
                    nx = importlib.import_module("networkx")
            except Exception:
                nx = None

            plt.switch_backend('Agg')

            config = test_result.get('test_config', {})
            topology_type = config.get('topology_type', test_result.get('topology_type', 'chain'))
            num_agents = config.get('num_agents', test_result.get('agent_count', 5))

            malicious_agents = test_result.get('malicious_agents', [])
            if not malicious_agents:

                malicious_agents = config.get('malicious_agents', [])
            if not malicious_agents and 'metadata' in test_result:
                malicious_agents = test_result['metadata'].get('malicious_agents', [])

            fig, ax = plt.subplots(1, 1, figsize=(12, 8))

            topology = self._create_topology_structure(topology_type, num_agents)

            agent_type = str(config.get('agent_type', test_result.get('method_type', '')))
            dataset_type = str(config.get('dataset_type', test_result.get('dataset_type', '')))
            if nx is not None:
                self._draw_networkx_topology(
                    ax, topology, malicious_agents, topology_type, num_agents,
                    agent_type=agent_type, dataset_type=dataset_type, nx_module=nx
                )
            else:

                positions = self._generate_node_positions(topology_type, num_agents)
                self._draw_topology_connections(ax, positions, topology_type)
                self._draw_topology_nodes(ax, positions, malicious_agents)
                self._add_topology_legend(ax)

            output_path = os.path.join(result_dir, 'topology_structure.png')
            plt.tight_layout()
            plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                       facecolor='white', edgecolor='none')
            plt.close()

            print(f"拓扑结构图已生成: {output_path}")

        except Exception as e:
            print(f"拓扑结构图生成失败: {e}")
            import traceback
            traceback.print_exc()

    def _generate_node_positions(self, topology_type: str, num_agents: int):

        positions = []

        if topology_type.lower() == 'chain':

            for i in range(num_agents):
                x = -0.8 + (1.6 * i / (num_agents - 1)) if num_agents > 1 else 0
                y = 0
                positions.append((x, y))

        elif topology_type.lower() == 'star':

            positions.append((0, 0))        
            for i in range(1, num_agents):
                angle = 2 * np.pi * (i - 1) / (num_agents - 1)
                x = 0.7 * np.cos(angle)
                y = 0.7 * np.sin(angle)
                positions.append((x, y))

        elif topology_type.lower() == 'ring':

            for i in range(num_agents):
                angle = 2 * np.pi * i / num_agents
                x = 0.7 * np.cos(angle)
                y = 0.7 * np.sin(angle)
                positions.append((x, y))

        else:

            for i in range(num_agents):
                angle = 2 * np.pi * i / num_agents
                x = 0.7 * np.cos(angle)
                y = 0.7 * np.sin(angle)
                positions.append((x, y))

        return positions

    def _draw_topology_connections(self, ax, positions, topology_type: str):

        num_nodes = len(positions)
        connection_color = '#2196F3'         

        if topology_type.lower() == 'chain':

            for i in range(num_nodes - 1):
                x1, y1 = positions[i]
                x2, y2 = positions[i + 1]
                ax.plot([x1, x2], [y1, y2], color=connection_color, 
                       linewidth=2, alpha=0.7)

        elif topology_type.lower() == 'star':

            center_x, center_y = positions[0]
            for i in range(1, num_nodes):
                x, y = positions[i]
                ax.plot([center_x, x], [center_y, y], color=connection_color, 
                       linewidth=2, alpha=0.7)

        elif topology_type.lower() == 'ring':

            for i in range(num_nodes):
                x1, y1 = positions[i]
                x2, y2 = positions[(i + 1) % num_nodes]
                ax.plot([x1, x2], [y1, y2], color=connection_color, 
                       linewidth=2, alpha=0.7)

    def _draw_topology_nodes(self, ax, positions, malicious_agents):

        import matplotlib.patches as patches

        normal_color = '#4CAF50'                 
        malicious_color = '#F44336'              

        for i, (x, y) in enumerate(positions):

            agent_id = f"agent_{i}"

            is_malicious = False
            if malicious_agents:

                is_malicious = agent_id in malicious_agents

                if not is_malicious:
                    for mal_agent in malicious_agents:
                        if str(mal_agent) == agent_id or str(mal_agent) == str(i):
                            is_malicious = True
                            break

            color = malicious_color if is_malicious else normal_color

            circle = patches.Circle((x, y), 0.08, facecolor=color, 
                                  edgecolor='black', linewidth=2)
            ax.add_patch(circle)

            ax.text(x, y-0.15, f"Node {i}", ha='center', va='top', 
                   fontsize=10, fontweight='bold')

            if is_malicious:
                ax.text(x, y, "X", ha='center', va='center', 
                       fontsize=12, color='white', fontweight='bold')
            else:
                ax.text(x, y, "O", ha='center', va='center', 
                       fontsize=12, color='white', fontweight='bold')

    def _add_topology_legend(self, ax):

        import matplotlib.patches as patches

        legend_elements = [
            patches.Patch(color='#4CAF50', label='正常节点'),
            patches.Patch(color='#F44336', label='恶意节点'),
            patches.Patch(color='#2196F3', label='网络连接')
        ]

        ax.legend(handles=legend_elements, loc='upper right', 
                 bbox_to_anchor=(1, 1))

    def _create_topology_structure(self, topology_type: str, num_agents: int) -> dict:

        topology = {}

        for i in range(num_agents):
            topology[f"agent_{i}"] = []

        if topology_type.lower() == 'chain':

            for i in range(num_agents):
                agent_id = f"agent_{i}"
                if i > 0:
                    topology[agent_id].append(f"agent_{i-1}")
                if i < num_agents - 1:
                    topology[agent_id].append(f"agent_{i+1}")

        elif topology_type.lower() == 'star':

            center = "agent_0"
            for i in range(1, num_agents):
                agent_id = f"agent_{i}"
                topology[center].append(agent_id)
                topology[agent_id].append(center)

        elif topology_type.lower() == 'complete':

            for i in range(num_agents):
                agent_i = f"agent_{i}"
                for j in range(num_agents):
                    if i != j:
                        agent_j = f"agent_{j}"
                        topology[agent_i].append(agent_j)

        else:

            for i in range(num_agents):
                agent_id = f"agent_{i}"
                if i > 0:
                    topology[agent_id].append(f"agent_{i-1}")
                if i < num_agents - 1:
                    topology[agent_id].append(f"agent_{i+1}")

        return topology

    def _draw_networkx_topology(self, ax, topology: dict, malicious_agents: list, 
                               topology_type: str, num_agents: int,
                               agent_type: str = "", dataset_type: str = "",
                               nx_module=None):

        import matplotlib.pyplot as plt
        nx = nx_module
        if nx is None:

            positions = self._generate_node_positions(topology_type, num_agents)
            self._draw_topology_connections(ax, positions, topology_type)
            self._draw_topology_nodes(ax, positions, malicious_agents)
            self._add_topology_legend(ax)
            ax.set_title('Byzantine Fault Tolerance Topology', fontsize=16, fontweight='bold', pad=20)
            ax.axis('off')
            return

        G = nx.Graph()

        all_nodes = set(topology.keys())
        for neighbors in topology.values():
            all_nodes.update(neighbors)

        for node in all_nodes:
            G.add_node(node)

        edges_added = set()
        for agent_id, neighbors in topology.items():
            for neighbor in neighbors:
                edge = tuple(sorted([agent_id, neighbor]))
                if edge not in edges_added and agent_id != neighbor:
                    G.add_edge(agent_id, neighbor)
                    edges_added.add(edge)

        num_nodes = G.number_of_nodes()
        if num_nodes <= 5:
            pos = nx.circular_layout(G)
        elif num_nodes <= 10:
            pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
        else:
            pos = nx.kamada_kawai_layout(G)

        nx.draw_networkx_edges(G, pos, edge_color='#757575', width=2, alpha=0.6)

        agent_type_l = str(agent_type).lower()
        dataset_type_l = str(dataset_type).lower()

        if 'decoder' in agent_type_l:
            is_llm_run = False
        else:
            is_llm_run = (
                ('llm' in agent_type_l) or
                ('probe' in agent_type_l) or
                ('safe_agents' in agent_type_l) or
                (dataset_type_l == 'safe' and 'traditional' not in agent_type_l)
            )

        for node in G.nodes():
            x, y = pos[node]

            if node in malicious_agents:
                node_color = '#F44336'      
                edge_color = 'darkred'
                node_label = f"{node}\n(Malicious)"
            else:
                node_color = '#4CAF50'      
                edge_color = 'darkgreen'
                node_label = f"{node}\n(Normal)"

            marker = 'o' if is_llm_run else 's'

            ax.scatter(x, y, c=node_color, s=800, marker=marker,
                      edgecolors=edge_color, linewidth=3, alpha=0.8)

            ax.text(x, y-0.15, node_label, ha='center', va='top', fontsize=9, 
                   fontweight='bold', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='white', alpha=0.8))

        legend_marker = 'o' if is_llm_run else 's'
        legend_elements = [
            plt.Line2D([0], [0], marker=legend_marker, color='w', markerfacecolor='#4CAF50',
                      markersize=12, label='Normal Node', markeredgecolor='darkgreen', markeredgewidth=2),
            plt.Line2D([0], [0], marker=legend_marker, color='w', markerfacecolor='#F44336',
                      markersize=12, label='Malicious Node', markeredgecolor='darkred', markeredgewidth=2)
        ]

        ax.legend(handles=legend_elements, loc='upper right', bbox_to_anchor=(1.15, 1))

        stats_text = f"Nodes: {G.number_of_nodes()}\nEdges: {G.number_of_edges()}\n"
        if G.number_of_nodes() > 0:
            avg_degree = 2 * G.number_of_edges() / G.number_of_nodes()
            stats_text += f"Avg Degree: {avg_degree:.1f}\nMalicious: {len(malicious_agents)}/{G.number_of_nodes()}"

        ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
               verticalalignment='top', bbox=dict(boxstyle="round,pad=0.5", 
               facecolor='lightgray', alpha=0.8))

        ax.set_title('Byzantine Fault Tolerance Topology', fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python comprehensive_analyzer.py <json结果文件> <输出txt文件>")
    else:
        import json
        import os

        json_path = sys.argv[1]
        output_path = sys.argv[2]

        with open(json_path, 'r', encoding='utf-8') as f:
            test_result = json.load(f)

        analyzer = ComprehensiveAnalyzer()
        result_dir = os.path.dirname(output_path)
        analyzer.analyze_test_result(test_result, result_dir)