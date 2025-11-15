#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_runner import BaseMethodRunner, REPO_ROOT
from ..interfaces import MethodType, IVisualizer
from ..visualization.unified_visualizer_adapter import UnifiedVisualizerAdapter
from config.prompt_probe_config import PromptProbeConfig

logger = logging.getLogger(__name__)

class PromptProbeRunner(BaseMethodRunner):

    def __init__(self, config: PromptProbeConfig):
        super().__init__(config)
        logger.info("初始化Prompt Probe运行器")

    def _get_method_type(self) -> MethodType:

        return MethodType.PROMPT_PROBE

    def _get_data_path(self) -> str:

        if hasattr(self.config, 'data_path') and self.config.data_path:
            return str(self.config.data_path)

        dataset_type = str(getattr(self.config, 'dataset_type', 'gsm8k')).lower()
        if dataset_type == 'safe':
            default_safe = REPO_ROOT / "data/byzantine/safe/safe_final_dataset_aligned_20250723_191623.json"
            return str(default_safe)
        elif dataset_type == 'commonsense':
            default_commonsense = REPO_ROOT / "data/byzantine/commonsense/gpt4omini_win_10.json"
            return str(default_commonsense)
        else:
            default_gsm8k = REPO_ROOT / "data/byzantine/gsm8k/gsm8k_final_dataset_20250723_154643.json"
            return str(default_gsm8k)

    def _get_data_kwargs(self) -> Dict[str, Any]:

        kwargs = {}

        if hasattr(self.config, 'dataset_type'):
            kwargs['dataset_type'] = str(self.config.dataset_type).lower()

        if hasattr(self.config, 'max_questions') and self.config.max_questions:
            kwargs['max_questions'] = self.config.max_questions

        if hasattr(self.config, 'question_types') and self.config.question_types:
            kwargs['question_types'] = self.config.question_types

        if getattr(self.config, 'use_hardcoded_questions', False):
            kwargs['use_hardcoded'] = True
            kwargs['hardcoded_questions'] = getattr(self.config, 'test_questions', [])

        return kwargs

    async def _create_visualizer(self) -> Optional[IVisualizer]:

        if not getattr(self.config, 'visualize', True):
            return None
        try:
            output_dir = getattr(self.config, 'visualizations_dir', getattr(self.config, 'output_dir', 'results'))
            return UnifiedVisualizerAdapter(
                output_dir,
                str(getattr(self.config, 'dataset_type', 'gsm8k')),
                str(getattr(self.config, 'agent_type', 'llm')),
            )
        except Exception as e:
            logger.warning(f"创建可视化器失败: {e}")
            return None

    async def _setup_malicious_agents(self) -> None:

        if self.config.malicious <= 0:
            logger.info("无恶意智能体")
            return

        logger.debug("设置Prompt Probe恶意智能体...")

        malicious_behavior = getattr(self.config, 'malicious_behavior', 'random')

        await super()._setup_malicious_agents()

        malicious_agents = [aid for aid, agent in self.agents.items() if agent.is_malicious]

        for agent_id in malicious_agents:
            agent = self.agents[agent_id]
            if hasattr(agent, 'set_malicious_behavior'):
                agent.set_malicious_behavior(malicious_behavior)

        logger.info(f"Prompt Probe恶意智能体设置完成: {malicious_agents}, 行为模式: {malicious_behavior}")

    def _serialize_config(self) -> Dict[str, Any]:

        config_dict = super()._serialize_config()

        config_dict.update({
            "method": "prompt_probe",
            "model_name": getattr(self.config, 'model', 'unknown'),
            "temperature": getattr(self.config, 'temperature', 0.7),
            "max_tokens": getattr(self.config, 'max_tokens', 512),
            "confidence_threshold": getattr(self.config, 'confidence_threshold', 0.8),
            "probe_type": getattr(self.config, 'probe_type', 'standard'),
            "use_hardcoded_questions": getattr(self.config, 'use_hardcoded_questions', True),
            "malicious_behavior": getattr(self.config, 'malicious_behavior', 'random'),
            "supported_topologies": getattr(self.config, 'supported_topologies', []),
            "position_strategies": getattr(self.config, 'position_strategies', [])
        })

        return config_dict

    async def _validate_prompt_probe_setup(self) -> None:

        if hasattr(self.config, 'confidence_threshold'):
            if not (0.0 <= self.config.confidence_threshold <= 1.0):
                raise ValueError(f"置信度阈值必须在0-1之间: {self.config.confidence_threshold}")

        if hasattr(self.config, 'supported_topologies'):
            current_topology = self.config.topology
            if current_topology not in self.config.supported_topologies:
                logger.warning(f"当前拓扑 {current_topology} 可能不被Prompt Probe完全支持")

        if not self.questions:
            if hasattr(self.config, 'test_questions') and self.config.test_questions:
                logger.info("使用配置中的硬编码测试问题")
            else:
                raise ValueError("未找到有效的问题数据")

        logger.debug("Prompt Probe设置验证通过")

    async def setup_experiment(self, config: Any = None) -> None:

        await super().setup_experiment(config)

        await self._validate_prompt_probe_setup()

def create_prompt_probe_runner(config: PromptProbeConfig) -> PromptProbeRunner:

    return PromptProbeRunner(config)

async def run_prompt_probe_experiment(config: PromptProbeConfig) -> Any:

    runner = create_prompt_probe_runner(config)

    try:
        result = await runner.run_experiment()
        return result
    finally:
        runner.cleanup_experiment()

if __name__ == "__main__":
    import asyncio
    from ...config.prompt_probe_config import parse_prompt_probe_args

    config = parse_prompt_probe_args()

    async def main():
        result = await run_prompt_probe_experiment(config)
        print(f"实验完成: {result.experiment_id}")
        print(f"准确率: {result.evaluation_metrics.get('overall_assessment', {}).get('accuracy', 'N/A')}")
        print(f"平均置信度: {result.evaluation_metrics.get('overall_assessment', {}).get('avg_confidence', 'N/A')}")

    asyncio.run(main())
