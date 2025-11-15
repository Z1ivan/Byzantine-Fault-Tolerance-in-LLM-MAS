#!/usr/bin/env python3

from datetime import datetime
from typing import Any, Dict, Tuple

def _norm_str(val: Any, default: str = "unknown") -> str:
    if val is None:
        return default
    try:
        s = str(val)
        return s
    except Exception:
        return default

def build_results_dir(base_root: str, method_name: str, dataset: str, agent_type: str,
                      topology: str, agents: int, malicious: int) -> str:
    group = f"{topology}_{agents}_{malicious}"
    return f"{base_root}/{method_name}/{dataset}/{agent_type}/{group}"

def generate_experiment_prefix(method_name: str, dataset: str, agent_type: str,
                               topology: str, agents: int, malicious: int,
                               timestamp: str = None) -> str:
    ts = timestamp or datetime.now().strftime('%Y%m%d_%H%M%S')
    return f"{method_name}_{dataset}_{agent_type}_{topology}_{agents}_{malicious}_{ts}"

def infer_from_config_or_result(config: Dict[str, Any], fallback: Dict[str, Any] = None,
                                method_name: str = 'pilot', base_root: str = None) -> Tuple[str, str]:

    fb = fallback or {}
    meta_cfg = (fb.get('metadata') or {}).get('config', {}) if isinstance(fb, dict) else {}

    dataset = (config.get('dataset_type') if isinstance(config, dict) else None) or \
              (fb.get('dataset_type') if isinstance(fb, dict) else None) or \
              meta_cfg.get('dataset_type')
    if not dataset:

        try:
            qs = fb.get('questions', []) if isinstance(fb, dict) else []
            if qs:
                is_safe = any(
                    str(q.get('correct_answer', '')).strip().lower() in ('safe', 'unsafe') or
                    'safety' in str(q.get('question_type', '')).lower()
                    for q in qs
                )
                dataset = 'safe' if is_safe else 'gsm8k'
        except Exception:
            dataset = 'gsm8k'
    dataset = _norm_str(dataset or 'gsm8k').lower()

    agent_type = (config.get('agent_type') if isinstance(config, dict) else None) or \
                 (fb.get('agent_type') if isinstance(fb, dict) else None) or \
                 meta_cfg.get('agent_type')
    if not agent_type:

        mt = None
        try:
            mt = (fb.get('method_type') if isinstance(fb, dict) else None) or meta_cfg.get('method')
        except Exception:
            mt = None
        agent_type = 'decoder' if str(mt).lower() in ('decoder', 'decoder_probe', 'hcp') else 'llm'
    agent_type = _norm_str(agent_type or 'llm').lower()

    topology = (config.get('topology_type') if isinstance(config, dict) else None) or \
               (fb.get('topology_type') if isinstance(fb, dict) else None) or \
               (fb.get('topology') if isinstance(fb, dict) else None) or \
               meta_cfg.get('topology')
    topology = _norm_str(topology or 'unknown')

    agents = config.get('num_agents') if isinstance(config, dict) else None
    if agents is None and isinstance(fb, dict):
        agents = fb.get('agent_count', fb.get('agents', None))
    if agents is None:
        agents = meta_cfg.get('agents', 0)
    agents = int(agents or 0)

    malicious = config.get('num_malicious') if isinstance(config, dict) else None
    if malicious is None and isinstance(fb, dict):
        malicious = fb.get('malicious_count', fb.get('malicious', None))
    if malicious is None:
        malicious = meta_cfg.get('malicious', 0)
    malicious = int(malicious or 0)

    prefix = generate_experiment_prefix(method_name, dataset, agent_type, topology, agents, malicious)
    output_dir = None
    if base_root:
        output_dir = build_results_dir(base_root, method_name, dataset, agent_type, topology, agents, malicious)
    return output_dir, prefix

