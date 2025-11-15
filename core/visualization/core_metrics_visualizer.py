#!/usr/bin/env python3

import os
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Any, Tuple
import re                              

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False

def convert_detailed_data_to_round_format(test_result: Dict[str, Any]) -> Dict[str, Any]:

    academic_evaluation = test_result.get('academic_evaluation', {})
    round_details = test_result.get('round_details', [])
    test_metadata = test_result.get('test_metadata', {})
    test_config = test_metadata.get('test_config', {})

    agent_responses = academic_evaluation.get('agent_responses', {})
    correct_answers = academic_evaluation.get('correct_answers', [])
    malicious_agents = academic_evaluation.get('malicious_agents', [])

    round_results = []

    for i, correct_answer in enumerate(correct_answers):

        question_info = {
            'question': f'Question {i+1}',
            'answer': str(correct_answer),                   
            'question_id': f'q_{i+1}',
            'label': test_config.get('dataset_type', 'unknown')
        }

        agent_answers = {}
        agent_results = []

        for agent_id, responses in agent_responses.items():
            if i < len(responses):
                initial_answer = str(responses[i])
                agent_answers[agent_id] = initial_answer

                agent_results.append({
                    'agent_id': agent_id,
                    'initial_answer': initial_answer,
                    'final_answer': initial_answer,                    
                    'is_malicious': agent_id in malicious_agents
                })

        round_data = {
            'round_number': i + 1,
            'question': question_info,
            'agent_answers': agent_answers,
            'agent_results': agent_results,
            'is_correct': True            
        }

        round_results.append(round_data)

    converted_result = {
        'round_results': round_results,
        'test_config': {
            'topology_type': test_config.get('topology_type', 'unknown'),
            'agent_type': 'TRADITIONAL_AGENTS',
            'num_agents': test_config.get('num_agents', len(agent_responses)),
            'num_malicious': test_config.get('num_malicious', len(malicious_agents)),
            'total_rounds': len(correct_answers),
            'test_mode': 'single',
            'timestamp': test_metadata.get('timestamp', ''),
            'dataset_type': test_config.get('dataset_type', 'unknown')
        },
        'malicious_agents': malicious_agents,
        'topology': test_result.get('topology_details', {}).get('topology', {})
    }

    return converted_result

def convert_gsm8k_to_round_format(test_result: Dict[str, Any]) -> Dict[str, Any]:

    converted_result = test_result.copy()

    test_results = test_result.get('test_results', [])
    agent_stats = test_result.get('agent_stats', {})
    malicious_agents = test_result.get('malicious_agents', [])

    is_safe_format = False
    if test_results and 'agent_detailed' in test_results[0]:
        is_safe_format = True

    test_config = {
        'topology_type': test_result.get('test_config', {}).get('topology', 'complete'),
        'agent_type': 'SAFE_AGENTS' if is_safe_format else 'GSM8K_PROBE',
        'num_agents': len(agent_stats) if agent_stats else (len(malicious_agents) + 1),
        'num_malicious': len(malicious_agents),
        'total_rounds': len(test_results),
        'test_mode': 'single',
        'timestamp': test_result.get('timestamp', ''),
        'malicious_agents': malicious_agents                    
    }

    def extract_answer_from_string(answer_str: str) -> str:

        if isinstance(answer_str, (int, float)):
            return str(answer_str)
        if isinstance(answer_str, str):

            match = re.search(r'answer:\s*(.+)', answer_str, re.IGNORECASE)
            if match:
                extracted = match.group(1).strip()

                if "," in extracted:
                    extracted = extracted.split(",")[0].strip()

                if "_INCOMPLETE" in extracted:

                    num_match = re.search(r'(\d+)_INCOMPLETE', extracted)
                    if num_match:
                        return num_match.group(1)
                    return "INCOMPLETE"

                return extracted

            if "_INCOMPLETE" in answer_str:
                num_match = re.search(r'(\d+)_INCOMPLETE', answer_str)
                if num_match:
                    return num_match.group(1)
                return "INCOMPLETE"

            return answer_str.strip()
        return str(answer_str)

    round_results = []
    for i, test_data in enumerate(test_results):
        if is_safe_format:

            question = test_data.get('question', {})
            agent_detailed = test_data.get('agent_detailed', {})

            question_text = question.get('question', f'Question {i+1}')
            question_id = question.get('id', f'Q{i+1}')

            question_label = question.get('label', 'safe')
            correct_answer = 'safe'

            adapted_question = {
                'question': question_text,
                'answer': correct_answer,
                'question_id': str(question_id)
            }

            agent_answers = {}
            consensus_results = {}
            agent_results = []

            for agent_id, agent_data in agent_detailed.items():

                raw_response = agent_data.get('raw_response', '')
                if 'answer:' in raw_response:
                    initial_token = raw_response.split('answer:')[1].split(',')[0].strip()
                else:
                    initial_token = '0'
                initial_answer = 'safe' if str(initial_token) == '1' else 'unsafe'

                final_token = str(agent_data.get('answer', '0'))
                final_answer = 'safe' if final_token == '1' else 'unsafe'
                confidence = agent_data.get('confidence', 0.5)
                is_malicious = agent_data.get('is_malicious', False)

                agent_answers[agent_id] = initial_answer          
                consensus_results[agent_id] = final_answer

                initial_is_correct = (initial_answer == correct_answer)
                final_is_correct = (final_answer == correct_answer)

                agent_results.append({
                    'agent_id': agent_id,
                    'answer': final_answer,
                    'confidence': confidence,
                    'is_correct': final_is_correct,
                    'is_malicious': is_malicious,
                    'initial_answer': initial_answer,
                    'initial_is_correct': initial_is_correct
                })
        else:

            question = test_data.get('question', {})
            agent_initial_results = test_data.get('agent_initial_results', {})
            agent_consensus_results = test_data.get('agent_consensus_results', {})
            consensus_evaluation = test_data.get('consensus_evaluation', {})

            question_text = question.get('question', f'Question {i+1}')
            correct_answer = consensus_evaluation.get('correct_answer', '')

            adapted_question = {
                'question': question_text,
                'answer': correct_answer,
                'question_id': question.get('question_id', f'Q{i+1}')
            }

            agent_answers = {}
            consensus_results = {}
            agent_results = []

            for agent_id, initial_data in agent_initial_results.items():

                if isinstance(initial_data, dict):
                    initial_answer_raw = str(initial_data.get('answer', ''))
                    initial_confidence = initial_data.get('confidence', 1.0)
                else:
                    initial_answer_raw = str(initial_data)
                    initial_confidence = 1.0

                initial_answer = extract_answer_from_string(initial_answer_raw)

                consensus_data = agent_consensus_results.get(agent_id, {})
                if isinstance(consensus_data, dict):
                    consensus_answer_raw = str(consensus_data.get('answer', initial_answer))
                else:
                    consensus_answer_raw = str(consensus_data) if consensus_data else initial_answer

                consensus_answer = extract_answer_from_string(consensus_answer_raw)

                agent_answers[agent_id] = initial_answer
                consensus_results[agent_id] = consensus_answer

                initial_is_correct = (initial_answer == correct_answer) if correct_answer else False
                consensus_is_correct = (consensus_answer == correct_answer) if correct_answer else False

                agent_results.append({
                    'agent_id': agent_id,
                    'answer': consensus_answer,
                    'confidence': initial_confidence,
                    'is_correct': consensus_is_correct,
                    'is_malicious': agent_id in malicious_agents,
                    'initial_answer': initial_answer,
                    'initial_is_correct': initial_is_correct,
                    'raw_response': initial_data.get('raw_response', '') if isinstance(initial_data, dict) else ''
                })

            final_consensus_raw = consensus_evaluation.get('final_consensus_answer', '')
            if not final_consensus_raw:
                final_consensus_raw = consensus_evaluation.get('consensus_answer', '')
            if not final_consensus_raw and consensus_results:
                final_consensus_raw = str(list(consensus_results.values())[0])

            final_consensus = extract_answer_from_string(final_consensus_raw)

            consensus_is_correct = (final_consensus == correct_answer) if correct_answer else False

        round_data = {
            'round_number': i + 1,
            'question': adapted_question,
            'agent_answers': agent_answers,
            'consensus_results': consensus_results,
            'agent_results': agent_results,
            'malicious_agents': malicious_agents,
            'consensus_result': final_consensus if not is_safe_format else list(consensus_results.values())[0] if consensus_results else 'safe',
            'is_correct': consensus_is_correct if not is_safe_format else True
        }

        round_results.append(round_data)

    topology = test_result.get('topology', {})
    if not topology:

        topology = test_result.get('topology_connections', {})
    if not topology:

        topology = {}
        all_agents = list(agent_stats.keys())
        for agent_id in all_agents:
            neighbors = [other_id for other_id in all_agents if other_id != agent_id]
            topology[agent_id] = neighbors

    successful_rounds = sum(1 for round_data in round_results if round_data.get('is_correct', False))
    success_rate = successful_rounds / len(round_results) if round_results else 0.0

    adapted_data = {
        'test_config': test_config,
        'round_results': round_results,
        'agent_stats': agent_stats,
        'malicious_agents': malicious_agents,
        'topology': topology,
        'summary': {
            'total_rounds': len(round_results),
            'successful_rounds': successful_rounds,
            'success_rate': success_rate
        }
    }

    return adapted_data

def create_core_metrics_visualization(test_result: Dict[str, Any], output_dir: str = None) -> str:

    if 'academic_evaluation' in test_result and 'round_details' in test_result:

        converted_result = convert_detailed_data_to_round_format(test_result)
    elif 'test_results' in test_result and 'round_results' not in test_result:

        converted_result = convert_gsm8k_to_round_format(test_result)
    else:

        if 'test_config' not in test_result or 'round_results' not in test_result:

            qs = test_result.get('questions', [])
            crs = test_result.get('consensus_results', [])
            agent_count = test_result.get('agent_count', 0)
            mal_count = test_result.get('malicious_count', 0)

            md = test_result.get('metadata', {}) or {}
            malicious_agents = md.get('malicious_agents', [])
            round_results = []
            for q, cr in zip(qs, crs):
                agent_answers = {}
                agent_results = []
                consensus_results = {}
                correct_answer = str(q.get('correct_answer', '')).strip()
                for r in cr.get('individual_responses', []):
                    aid = r.get('agent_id')
                    ans = str(r.get('answer'))
                    agent_answers[aid] = ans
                    consensus_results[aid] = ans
                    agent_results.append({
                        'agent_id': aid,
                        'answer': ans,
                        'confidence': float(r.get('confidence', 0.0)),
                        'is_correct': (ans == correct_answer) if correct_answer else False,
                        'initial_answer': ans,
                        'initial_is_correct': (ans == correct_answer) if correct_answer else False
                    })
                round_results.append({
                    'round_number': cr.get('round_number', 1),
                    'question': { 'question': q.get('question_text',''), 'answer': correct_answer, 'question_id': q.get('question_id','') },
                    'agent_answers': agent_answers,
                    'consensus_results': consensus_results,
                    'agent_results': agent_results,
                    'malicious_agents': malicious_agents,
                    'consensus_result': cr.get('consensus_answer'),
                    'is_correct': (str(cr.get('consensus_answer')) == correct_answer) if correct_answer else False
                })

            def detect_dataset_type(questions):
                if any(str(q.get('question_type','')).startswith('safety') or 
                       str(q.get('correct_answer','')).lower() in ('safe','unsafe') for q in questions):
                    return 'safe'

                if any(q.get('metadata', {}).get('choices') and 
                       len(q.get('metadata', {}).get('choices', [])) == 5 for q in questions):
                    return 'commonsense'

                if any('what' in str(q.get('question_text', '')).lower() or
                       'where' in str(q.get('question_text', '')).lower() or
                       'who' in str(q.get('question_text', '')).lower() for q in questions):

                    if any(not str(q.get('correct_answer', '')).replace('.','').replace('-','').isdigit() 
                           for q in questions if q.get('correct_answer')):
                        return 'commonsense'
                return 'gsm8k'

            converted_result = {
                'test_config': {
                    'topology_type': test_result.get('topology_type','unknown'),
                    'agent_type': test_result.get('method_type','decoder_probe'),
                    'num_agents': agent_count,
                    'num_malicious': mal_count,
                    'total_rounds': len(round_results),
                    'dataset_type': detect_dataset_type(qs),
                    'malicious_agents': malicious_agents
                },
                'round_results': round_results,
                'agent_stats': { aid: {'agent_type': test_result.get('method_type','decoder_probe')} for aid in {r['agent_id'] for cr in test_result.get('consensus_results', []) for r in cr.get('individual_responses', [])} },
                'malicious_agents': malicious_agents,
                'topology': {}
            }
        else:

            converted_result = test_result

    if output_dir is None:
        output_dir = "."

    os.makedirs(output_dir, exist_ok=True)

    from .metrics.node_accuracy import NodeAccuracyCalculator
    from .metrics.consensus_accuracy import ConsensusAccuracyCalculator
    from .metrics.msbe_calculator import MSBECalculator
    from .metrics.consensus_error import ConsensusErrorCalculator
    from .metrics.base_calculator import TestData

    test_data = TestData(
        config=converted_result.get('test_config', {}),
        rounds=converted_result.get('round_results', []),
        malicious_agents=converted_result.get('malicious_agents', []),
        topology=converted_result.get('topology', {})
    )

    node_calc = NodeAccuracyCalculator()
    node_metrics = node_calc.calculate(test_data)

    if 'summary' in test_result and 'consensus_accuracy' in test_result['summary']:
        consensus_metrics = {
            'consensus_accuracy': test_result['summary']['consensus_accuracy'],
            'correct_consensus': test_result['summary'].get('consensus_correct_rounds', 0),
            'total_consensus': test_result['summary'].get('total_rounds', 0),
            'explanation': 'GSM8K数据：直接从实验结果summary中读取'
        }
    else:
        consensus_calc = ConsensusAccuracyCalculator()
        consensus_metrics = consensus_calc.calculate(test_data)

    msbe_calc = MSBECalculator()
    msbe_metrics = msbe_calc.calculate(test_data)

    ce_calc = ConsensusErrorCalculator()
    ce_metrics = ce_calc.calculate(test_data)

    config = converted_result['test_config']

    agent_type = config.get('agent_type')
    if not agent_type and 'agent_stats' in converted_result:

        first_agent = list(converted_result['agent_stats'].keys())[0]
        agent_type = converted_result['agent_stats'][first_agent].get('agent_type', 'UNKNOWN')

    dataset_type = str(config.get('dataset_type', '')).upper()
    agent_type_display = agent_type.upper() if agent_type else 'UNKNOWN'
    if 'SAFE' in dataset_type:
        agent_type_display = 'LLM+SAFE' if 'LLM' in agent_type_display or 'SAFE_AGENTS' in agent_type_display else f'{agent_type_display}+SAFE'

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Byzantine Fault Tolerance - Core Metrics ({agent_type_display})', 
                 fontsize=16, fontweight='bold')

    _plot_node_accuracy_new(ax1, node_metrics, config)

    _plot_consensus_accuracy_new(ax2, consensus_metrics)

    _plot_bellman_error_new(ax3, msbe_metrics)

    _plot_consensus_error_new(ax4, ce_metrics)

    plt.tight_layout()

    chart_path = os.path.join(output_dir, "core_metrics.png")
    try:
        cfg = (converted_result.get('test_config') or {})

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
        method_name = _infer_method_name(cfg, converted_result)
        from ..utils.naming import infer_from_config_or_result
        output_dir_tmp, prefix = infer_from_config_or_result(cfg, converted_result, method_name=method_name)
        chart_path = os.path.join(output_dir, f"{prefix}_core_metrics.png")
    except Exception:
        pass
    plt.savefig(chart_path, dpi=200, bbox_inches='tight', facecolor='white')
    plt.close()

    print(f"Generated core metrics visualization: {os.path.basename(chart_path)}")
    return output_dir

def generate_visualization_from_json(json_file_path: str):

    import json

    print(f"Loading JSON file: {json_file_path}")

    if not os.path.exists(json_file_path):
        print(f"File not found: {json_file_path}")
        return

    try:
        with open(json_file_path, 'r', encoding='utf-8') as f:
            test_result = json.load(f)
        print("JSON file loaded successfully")
    except Exception as e:
        print(f"Failed to load JSON: {e}")
        return

    output_dir = os.path.dirname(json_file_path)
    if not output_dir:                
        output_dir = "."

    try:
        create_core_metrics_visualization(test_result, output_dir)
        print(f"Core metrics visualization generated in: {output_dir}")

        chart_file = os.path.join(output_dir, "core_metrics.png")
        if os.path.exists(chart_file):
            file_size = os.path.getsize(chart_file) / 1024
            print(f"Generated: core_metrics.png ({file_size:.1f} KB)")

    except Exception as e:
        print(f"Visualization generation failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    import sys

    print("Core Metrics Visualizer")
    print("=" * 40)
    print("用法: python core_metrics_visualizer.py <json_file_path>")
    print("或者在Python中调用: generate_visualization_from_json(json_file_path)")

    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
        generate_visualization_from_json(json_file_path)
    else:
        print("\n示例:")
        print("python core_metrics_visualizer.py results/byzantine_test_xxx/xxx.json")

def _plot_node_accuracy_new(ax, node_metrics: Dict[str, Any], config: Dict[str, Any]):

    node_accuracies = node_metrics.get('node_accuracies', {})
    if not node_accuracies:
        ax.text(0.5, 0.5, 'No data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('1. Node Accuracy', fontweight='bold')
        return

    agents = sorted(node_accuracies.keys())
    accuracies = [node_accuracies[agent]['accuracy'] for agent in agents]
    malicious_agents = config.get('malicious_agents', [])
    malicious_set = set(malicious_agents) if isinstance(malicious_agents, list) else set()

    bars = ax.bar(range(len(agents)), accuracies, color='#28A745', alpha=0.8)
    ax.set_title('1. Node Accuracy', fontweight='bold')
    ax.set_xlabel('Agents')
    ax.set_ylabel('Accuracy Rate')
    ax.set_xticks(range(len(agents)))
    ax.set_xticklabels([f'A{i+1}' for i in range(len(agents))], rotation=45)
    ax.set_ylim(0, 1.1)

    for bar, acc in zip(bars, accuracies):
        y = max(bar.get_height(), 0.0)
        ax.text(bar.get_x() + bar.get_width()/2, y + 0.02,
                f'{acc:.2f}', ha='center', fontweight='bold', fontsize=10)

def _plot_consensus_accuracy_new(ax, consensus_metrics: Dict[str, Any]):

    accuracy = consensus_metrics.get('consensus_accuracy', 0)
    correct_count = consensus_metrics.get('correct_consensus', 0)
    total_count = consensus_metrics.get('total_consensus', 0)

    if total_count == 0:
        ax.text(0.5, 0.5, 'No consensus data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('2. Consensus Accuracy', fontweight='bold')
        return

    categories = ['Consensus\nAccuracy']
    rates = [accuracy]
    colors = ['#28A745']

    bars = ax.bar(categories, rates, color=colors, alpha=0.8, width=0.6)

    for bar, rate in zip(bars, rates):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{rate:.1%}', ha='center', fontweight='bold', fontsize=12)

    ax.set_ylim(0, 1.1)
    ax.set_ylabel('Consensus Success Rate', fontweight='bold')
    ax.set_title(f'2. Consensus Accuracy\n({correct_count}/{total_count} questions)', 
                fontweight='bold', fontsize=12)
    ax.grid(True, alpha=0.3, axis='y')

def _plot_bellman_error_new(ax, msbe_metrics: Dict[str, Any]):

    question_details = msbe_metrics.get('question_details', [])
    if not question_details:
        ax.text(0.5, 0.5, 'No MSBE data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('3. Mean Squared Bellman Error', fontweight='bold')
        return

    bellman_errors = [detail['msbe'] for detail in question_details]
    question_ids = [detail['question_id'] for detail in question_details]

    x = range(len(bellman_errors))
    ax.plot(x, bellman_errors, marker='o', linewidth=2, markersize=8, color='#FF6B35')
    ax.fill_between(x, bellman_errors, alpha=0.3, color='#FF6B35')

    ax.set_title('3. Mean Squared Bellman Error', fontweight='bold')
    ax.set_xlabel('Questions')
    ax.set_ylabel('MSE')
    ax.set_xticks(x)
    ax.set_xticklabels([f'Q{i+1}' for i in range(len(question_ids))], rotation=45)
    ax.grid(True, alpha=0.3)

    avg_error = msbe_metrics.get('mean_msbe', 0)
    ax.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8,
              label=f'Average: {avg_error:.3f}')
    ax.legend()

def _plot_consensus_error_new(ax, ce_metrics: Dict[str, Any]):

    question_details = ce_metrics.get('question_details', [])
    if not question_details:
        ax.text(0.5, 0.5, 'No CE data available', ha='center', va='center', transform=ax.transAxes)
        ax.set_title('4. Consensus Error', fontweight='bold')
        return

    consensus_errors = [detail['ce'] for detail in question_details]
    question_ids = [detail['question_id'] for detail in question_details]

    colors = ['#4169E1' if err < 0.5 else '#FFB000' for err in consensus_errors]
    bars = ax.bar(range(len(consensus_errors)), consensus_errors, color=colors, alpha=0.8)

    ax.set_title('4. Consensus Error', fontweight='bold')
    ax.set_xlabel('Questions')
    ax.set_ylabel('Error Rate')
    ax.set_xticks(range(len(question_ids)))
    ax.set_xticklabels([f'Q{i+1}' for i in range(len(question_ids))], rotation=45)
    ax.set_ylim(0, max(1.1, max(consensus_errors) * 1.1) if consensus_errors else 1.1)

    for bar, err in zip(bars, consensus_errors):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{err:.2f}', ha='center', fontweight='bold', fontsize=10)

    avg_error = ce_metrics.get('mean_ce', 0)
    ax.axhline(y=avg_error, color='red', linestyle='--', alpha=0.8,
              label=f'Average: {avg_error:.3f}')
    ax.legend()

    ax.grid(True, alpha=0.3) 