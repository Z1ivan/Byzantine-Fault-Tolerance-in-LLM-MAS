#!/usr/bin/env python3

import logging
from pathlib import Path
from typing import Dict, Any, Optional

from .base_runner import BaseMethodRunner, REPO_ROOT
from ..interfaces import MethodType, IVisualizer
from ..visualization.unified_visualizer_adapter import UnifiedVisualizerAdapter
from config.base_config import BaseConfig

logger = logging.getLogger(__name__)

class PilotRunner(BaseMethodRunner):

    def __init__(self, config: BaseConfig):
        super().__init__(config)
        logger.info("初始化先导实验运行器 (PilotRunner)")

    def _get_method_type(self) -> MethodType:

        return MethodType.PILOT

    def _get_data_path(self) -> str:

        dataset_type = getattr(self.config, 'dataset_type', 'gsm8k')

        explicit_path = None
        if hasattr(self.config, 'data_file') and self.config.data_file:
            explicit_path = str(self.config.data_file)
        elif hasattr(self.config, 'data_path') and self.config.data_path:
            explicit_path = str(self.config.data_path)

        default_safe = REPO_ROOT / "data/byzantine/safe/safe_final_dataset_aligned_20250723_191623.json"
        default_gsm8k = REPO_ROOT / "data/byzantine/gsm8k/gsm8k_final_dataset_20250723_154643.json"

        if dataset_type == 'safe':
            if not explicit_path:
                return str(default_safe)
            p = str(explicit_path)
            if p.endswith('byzantine_test_questions.json') or '/gsm8k/' in p:
                return str(default_safe)
            return p
        else:
            return explicit_path or str(default_gsm8k)

    def _get_data_kwargs(self) -> Dict[str, Any]:

        kwargs = {}
        if hasattr(self.config, 'dataset_type') and self.config.dataset_type:
            kwargs['dataset_type'] = self.config.dataset_type
        if hasattr(self.config, 'max_questions') and self.config.max_questions:
            kwargs['max_questions'] = self.config.max_questions
        if hasattr(self.config, 'question_filter') and self.config.question_filter:
            kwargs['question_filter'] = self.config.question_filter
        return kwargs

    async def _create_visualizer(self) -> Optional[IVisualizer]:

        if not getattr(self.config, 'visualize', True):
            return None
        try:
            output_dir = getattr(self.config, 'visualizations_dir', getattr(self.config, 'output_dir', 'results'))
            return UnifiedVisualizerAdapter(
                output_dir,
                str(getattr(self.config, 'dataset_type', 'gsm8k')),
                str(getattr(self.config, 'agent_type', 'traditional')),
            )
        except Exception as e:
            logger.warning(f"创建可视化器失败: {e}")
            return None

    def _serialize_config(self) -> Dict[str, Any]:

        config_dict = super()._serialize_config()
        config_dict.update({
            "method": "pilot",
            "model_name": getattr(self.config, 'model', 'unknown'),
            "temperature": getattr(self.config, 'temperature', 0.1),
            "max_tokens": getattr(self.config, 'max_tokens', 1024),
            "use_cot": getattr(self.config, 'use_cot', True),
            "probe_training_enabled": getattr(self.config, 'enable_probe_training', False)
        })
        return config_dict

def create_pilot_runner(config: BaseConfig) -> PilotRunner:

    return PilotRunner(config)

async def run_pilot_experiment(config: BaseConfig) -> Any:

    runner = create_pilot_runner(config)
    try:
        result = await runner.run_experiment()
        return result
    finally:
        runner.cleanup_experiment()

if __name__ == "__main__":
    import asyncio
    from ...config.base_config import create_base_config_from_args
    config = create_base_config_from_args()
    async def main():
        result = await run_pilot_experiment(config)
        print(f"实验完成: {result.experiment_id}")
        print(f"准确率: {result.evaluation_metrics.get('overall_assessment', {}).get('accuracy', 'N/A')}")
    asyncio.run(main())

