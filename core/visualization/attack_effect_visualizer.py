#!/usr/bin/env python3

import os
import json
import matplotlib.pyplot as plt
import numpy as np
from typing import Dict, List, Any, Tuple
import seaborn as sns
import matplotlib
import pandas as pd
from pathlib import Path
import logging
from datetime import datetime

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'sans-serif']
plt.rcParams['axes.unicode_minus'] = False            

matplotlib.rcParams['font.family'] = ['DejaVu Sans', 'sans-serif']

class AttackEffectAnalyzer:
    def __init__(self):
        self.colors = {
            'attack_success': '#DC3545',               
            'defense_success': '#28A745',              
            'partial_success': '#FFC107',              
            'neutral': '#6C757D',                    
            'danger_zone': '#FF6B6B',             
            'safe_zone': '#51CF66'                
        }

    def create_attack_effect_analysis(self, test_result: Dict[str, Any], output_dir: str = None) -> str:

        if output_dir is None:
            output_dir = "."

        os.makedirs(output_dir, exist_ok=True)

        config = test_result.get('test_config', {})
        if not config:

            config = {
                'agent_type': test_result.get('method_type', 'traditional'),
                'num_agents': test_result.get('agent_count', 5),
                'topology_type': test_result.get('topology_type', 'chain')
            }

        rounds = test_result.get('round_results', [])
        malicious_agents = test_result.get('malicious_agents', [])

        if not rounds and 'questions' in test_result and 'consensus_results' in test_result:
            rounds = self._convert_pilot_experiment_to_rounds(test_result)

        if not malicious_agents and 'metadata' in test_result:
            malicious_agents = test_result['metadata'].get('malicious_agents', [])

        if not rounds:
            print("No round data, cannot generate attack effect analysis")
            return None

        for i, round_data in enumerate(rounds):
            correct_answer = round_data['question']['answer']

            if not round_data.get('consensus_results') and round_data.get('agent_results'):

                round_data['consensus_results'] = {a['agent_id']: round_data.get('consensus_result', 0) for a in round_data['agent_results']}

        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

        agent_type = config.get('agent_type', 'unknown').upper()
        total_agents = config.get('num_agents', len(malicious_agents) + 5)      
        malicious_ratio = len(malicious_agents) / total_agents * 100

        fig.suptitle(
            f'Byzantine Attack Effect Analysis - {len(malicious_agents)}/{total_agents} Malicious ({malicious_ratio:.1f}%) - {agent_type} Agents',
            fontsize=14, fontweight='bold', y=0.98
        )

        self._plot_attack_success_rate(ax1, rounds, malicious_agents)

        self._plot_system_resistance(ax2, rounds, malicious_agents, total_agents)

        self._plot_contamination_spread(ax3, rounds, malicious_agents)

        self._plot_simplified_safety_status(ax4, malicious_agents, total_agents)

        plt.tight_layout(rect=[0, 0.02, 1, 0.96])           

        chart_path = os.path.join(output_dir, "attack_effect_analysis.png")
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
            from ..utils.naming import infer_from_config_or_result
            _dir, prefix = infer_from_config_or_result(cfg, test_result, method_name=method_name)
            chart_path = os.path.join(output_dir, f"{prefix}_attack_effect.png")
        except Exception:
            pass
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()

        print(f"Generated attack effect analysis chart: {os.path.basename(chart_path)}")
        return chart_path

    def _plot_attack_success_rate(self, ax, rounds, malicious_agents):

        question_results = []

        for i, round_data in enumerate(rounds):
            correct_answer = round_data['question']['answer']

            is_safe_dataset = (correct_answer in ['safe', 'unsafe'] or 
                             '_dataset_type' in round_data['question'] and 
                             round_data['question']['_dataset_type'] == 'safe')

            normal_correct = 0
            normal_total = 0
            malicious_wrong = 0
            malicious_total = 0

            individual_answers = round_data.get('agent_answers', {})
            consensus_answers = round_data.get('consensus_results', {})
            answers_data = consensus_answers if consensus_answers else individual_answers

            for agent, final_answer in answers_data.items():
                if is_safe_dataset:

                    final_str = str(final_answer).strip().lower()
                    if final_str in ('0','1'):
                        mapped_answer = 'safe' if final_str == '1' else 'unsafe'
                    else:
                        mapped_answer = final_str                  
                    is_correct = mapped_answer == str(correct_answer).strip().lower()
                else:

                    is_correct = str(final_answer).strip() == str(correct_answer).strip()

                if agent in malicious_agents:
                    malicious_total += 1
                    if not is_correct:
                        malicious_wrong += 1
                else:
                    normal_total += 1
                    if is_correct:
                        normal_correct += 1

            defense_success_rate = normal_correct / normal_total if normal_total > 0 else 0
            attack_execution_rate = malicious_wrong / malicious_total if malicious_total > 0 else 0

            if defense_success_rate > 0.8:
                attack_effect = 'Defense Success'
            elif defense_success_rate < 0.5:
                attack_effect = 'Attack Success'
            else:
                attack_effect = 'Partial Success'

            question_results.append({
                'question': f'Q{i+1}',
                'defense_rate': defense_success_rate,
                'attack_rate': attack_execution_rate,
                'effect': attack_effect
            })

        questions = [r['question'] for r in question_results]
        defense_rates = [r['defense_rate'] for r in question_results]
        attack_rates = [r['attack_rate'] for r in question_results]

        x = np.arange(len(questions))
        width = 0.35

        bars1 = ax.bar(x - width/2, defense_rates, width, 
                       label='Defense Success Rate', color=self.colors['defense_success'], alpha=0.8)

        bars2 = ax.bar(x + width/2, attack_rates, width, 
                       label='Attack Execution Rate', color=self.colors['attack_success'], alpha=0.8)

        ax.set_xlabel('Question', fontweight='bold')
        ax.set_ylabel('Success Rate', fontweight='bold')
        ax.set_title('1. Attack vs Defense Success Rate', fontweight='bold', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(questions)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_ylim(0, 1.1)

        for bar, value in zip(bars1, defense_rates):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.1%}', ha='center', va='bottom', fontsize=9)

        for bar, value in zip(bars2, attack_rates):
            if value > 0:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                       f'{value:.1%}', ha='center', va='bottom', fontsize=9)

    def _plot_system_resistance(self, ax, rounds, malicious_agents, total_agents):

        malicious_ratio = len(malicious_agents) / total_agents if total_agents > 0 else 0
        theoretical_limit = 1/3             

        resistance_factors = {
            'Network Topology': 0.67,
            'Consensus Algorithm': 0.40, 
            'Node Authentication': 0.27
        }

        factors = list(resistance_factors.keys())
        values = list(resistance_factors.values())
        colors = [self.colors['defense_success'], self.colors['attack_success'], self.colors['neutral']]

        bars = ax.barh(factors, values, color=colors, alpha=0.8)

        ax.set_title('2. System Resistance Analysis', fontweight='bold', fontsize=12)
        ax.set_xlim(0, 1.1)
        ax.grid(True, alpha=0.3, axis='x')

        for bar, value in zip(bars, values):
            width = bar.get_width()
            if width > 0:
                ax.text(width + 0.02, bar.get_y() + bar.get_height()/2.,
                       f'{value:.2f}', ha='left', va='center', fontweight='bold')

        if malicious_ratio > theoretical_limit:
            ax.text(0.95, 0.95, 'ALERT: Beyond Safety Threshold', transform=ax.transAxes,
                   ha='right', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='red', alpha=0.3))
        else:
            ax.text(0.95, 0.95, 'OK: Within Safety Range', transform=ax.transAxes,
                   ha='right', va='top', fontsize=12, fontweight='bold',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='green', alpha=0.3))

    def _plot_contamination_spread(self, ax, rounds, malicious_agents):

        questions = [f'Q{i+1}' for i in range(len(rounds))]
        contamination_data = []

        def normalize_answer(answer, correct_answer):

            answer_str = str(answer).strip().lower()
            correct_str = str(correct_answer).strip().lower()

            if correct_str in ['safe', 'unsafe']:
                if answer_str == '1':
                    return 'safe'
                elif answer_str == '0':
                    return 'unsafe'
                else:
                    return answer_str

            return answer_str

        for round_data in rounds:

            correct_answer = round_data.get('question', {}).get('answer', '')

            agent_answers = round_data.get('agent_answers', {})

            normal_agents = [agent for agent in agent_answers.keys() if agent not in malicious_agents]
            normal_total = len(normal_agents)

            contaminated = 0
            for agent in normal_agents:
                agent_answer = agent_answers.get(agent, '')
                normalized_agent = normalize_answer(agent_answer, correct_answer)
                normalized_correct = normalize_answer(correct_answer, correct_answer)
                if normalized_agent != normalized_correct:
                    contaminated += 1

            contamination_rate = contaminated / normal_total if normal_total > 0 else 0

            contamination_data.append({
                'contaminated': contaminated,
                'normal_total': normal_total,
                'rate': contamination_rate
            })

        contaminated_counts = [d['contaminated'] for d in contamination_data]
        safe_counts = [d['normal_total'] - d['contaminated'] for d in contamination_data]
        contamination_rates = [d['rate'] for d in contamination_data]

        bars1 = ax.bar(questions, safe_counts, label='Uncontaminated Nodes', 
               color=self.colors['defense_success'], alpha=0.8)
        bars2 = ax.bar(questions, contaminated_counts, bottom=safe_counts, 
               label='Contaminated Nodes', color=self.colors['attack_success'], alpha=0.8)

        ax.set_xlabel('Questions', fontweight='bold')
        ax.set_ylabel('Node Count', fontweight='bold')
        ax.set_title('3. Malicious Contamination Spread', fontweight='bold', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3, axis='y')

        ax2 = ax.twinx()
        line = ax2.plot(questions, contamination_rates, color='red', marker='o', 
                linewidth=3, markersize=8, label='Contamination Rate')
        ax2.set_ylabel('Contamination Rate', fontweight='bold', color='red')
        ax2.set_ylim(0, max(1, max(contamination_rates) * 1.2) if contamination_rates else 1)
        ax2.tick_params(axis='y', labelcolor='red')

        for i, (cont, rate) in enumerate(zip(contaminated_counts, contamination_rates)):
            if cont > 0:
                ax.text(i, safe_counts[i] + cont/2, str(cont), 
                       ha='center', va='center', fontweight='bold', color='white')

            ax2.text(i, rate + max(contamination_rates) * 0.05 if contamination_rates else 0.05, 
                    f'{rate:.1%}', ha='center', va='bottom', 
                    fontweight='bold', color='red', fontsize=9)

        avg_contamination = np.mean(contamination_rates) if contamination_rates else 0
        max_contamination = max(contamination_rates) if contamination_rates else 0
        ax.text(0.02, 0.98, f'Avg Contamination: {avg_contamination:.1%}\nMax Contamination: {max_contamination:.1%}', 
               transform=ax.transAxes, ha='left', va='top',
               bbox=dict(boxstyle="round,pad=0.3", facecolor='yellow', alpha=0.3),
               fontsize=9)

    def _plot_simplified_safety_status(self, ax, malicious_agents, total_agents):

        malicious_count = len(malicious_agents)

        theoretical_max = (total_agents - 1) // 3               

        if malicious_count <= theoretical_max:
            safety_status = "SAFE"
            safety_color = self.colors['defense_success']
            risk_level = "Low Risk"
        else:
            safety_status = "DANGER"  
            safety_color = self.colors['attack_success']
            risk_level = "High Risk"

        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 8, f'System Status: {safety_status}', 
                ha='center', va='center', fontsize=18, fontweight='bold',
                bbox=dict(boxstyle="round,pad=0.5", facecolor=safety_color, alpha=0.3))

        info_text = f"""
Network: {total_agents} nodes
Malicious: {malicious_count} nodes ({malicious_count/total_agents:.1%})
Theoretical Limit: {theoretical_max} nodes
Safety Margin: {theoretical_max - malicious_count} nodes
        """
        ax.text(5, 4.5, info_text.strip(), ha='center', va='center', 
               fontsize=12, bbox=dict(boxstyle="round,pad=0.5", facecolor='lightgray', alpha=0.3))

        risk_ratio = malicious_count / max(theoretical_max, 1)
        risk_width = min(risk_ratio * 8, 8)

        ax.barh(1.5, risk_width, height=0.5, left=1, 
               color=safety_color, alpha=0.7, label=f'Risk Level: {risk_ratio:.1%}')
        ax.barh(1.5, 8-risk_width, height=0.5, left=1+risk_width, 
               color='lightgray', alpha=0.3)

        ax.text(5, 0.8, f'Byzantine Fault Tolerance: {risk_ratio:.1%} of limit', 
               ha='center', va='center', fontsize=10, fontweight='bold')

        ax.set_title('4. System Safety Assessment', fontweight='bold', fontsize=12)

    def _convert_pilot_experiment_to_rounds(self, test_result: Dict[str, Any]) -> List[Dict]:

        questions = test_result.get('questions', [])
        consensus_results = test_result.get('consensus_results', [])
        rounds = []

        for idx, (question_data, consensus_data) in enumerate(zip(questions, consensus_results)):
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
                'is_correct': consensus_data.get('convergence_achieved', True)
            }

            if 'individual_responses' in consensus_data:
                for response in consensus_data['individual_responses']:
                    agent_id = response.get('agent_id')
                    answer = response.get('answer', '')
                    if agent_id:
                        round_result['agent_answers'][agent_id] = answer
                        round_result['consensus_results'][agent_id] = answer

            rounds.append(round_result)

        return rounds

def create_attack_effect_analysis(test_result: Dict[str, Any], output_dir: str = None) -> str:

    analyzer = AttackEffectAnalyzer()
    return analyzer.create_attack_effect_analysis(test_result, output_dir)

if __name__ == "__main__":
    import sys

    print("Byzantine Attack Effect Analyzer")
    print("=" * 50)
    print("Focuses on displaying attack success rates and system defense effectiveness")
    print("Core functions:")
    print("1. Attack vs Defense Success Rate Comparison")
    print("2. System Resistance Assessment")
    print("3. Malicious Contamination Spread Analysis")
    print("4. Safety Threshold Comparison Analysis")

    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]

        print(f"Analyzing attack effect: {json_file_path}")

        if not os.path.exists(json_file_path):
            print(f"File not found: {json_file_path}")
            sys.exit(1)

        try:
            with open(json_file_path, 'r', encoding='utf-8') as f:
                test_result = json.load(f)
            print("JSON file loaded successfully")
        except Exception as e:
            print(f"JSON loading failed: {e}")
            sys.exit(1)

        output_dir = os.path.dirname(json_file_path)
        if not output_dir:                
            output_dir = "."

        try:
            chart_path = create_attack_effect_analysis(test_result, output_dir)
            if chart_path:
                file_size = os.path.getsize(chart_path) / 1024
                print(f"Attack effect analysis chart generated successfully: {os.path.basename(chart_path)} ({file_size:.1f} KB)")

        except Exception as e:
            print(f"Attack effect analysis chart generation failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print("\nUsage: python attack_effect_visualizer.py <json_file_path>")
        print("Example: python attack_effect_visualizer.py results/byzantine_test_xxx/xxx.json")
