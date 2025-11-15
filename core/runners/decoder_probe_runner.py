#!/usr/bin/env python3

import logging
from typing import Dict, Any, Optional

from .base_runner import BaseMethodRunner, REPO_ROOT
from ..interfaces import MethodType, IVisualizer
from ..visualization.unified_visualizer_adapter import UnifiedVisualizerAdapter
from config.decoder_probe_config import DecoderProbeConfig

logger = logging.getLogger(__name__)

class DecoderProbeRunner(BaseMethodRunner):
    def __init__(self, config: DecoderProbeConfig):
        super().__init__(config)
        logger.info("初始化 Decoder Probe 运行器")

    def _get_method_type(self) -> MethodType:
        return MethodType.DECODER

    def _get_data_path(self) -> str:

        from pathlib import Path
        import os

        dataset = str(getattr(self.config, 'dataset_type', 'gsm8k')).lower()

        official_safe = REPO_ROOT / "data/byzantine/safe/all_llama31_win.json"
        official_gsm8k = REPO_ROOT / "data/byzantine/gsm8k/llama3.1_10.json"
        official_commonsense = REPO_ROOT / "data/byzantine/commonsense/llama31_win_10.json"

        try:
            dp = getattr(self.config, 'data_path', None)
            if dp:
                dp_path = Path(dp)
                if 'backup_0805' in str(dp_path):
                    if os.environ.get('BYZ_ALLOW_BACKUP_DATA', '1') not in ('1', 'true', 'True'):
                        logger.warning(f"检测到备份数据路径被传入，已改用正式数据集；若需使用备份数据请设置 BYZ_ALLOW_BACKUP_DATA=1: {dp_path}")

                        if dataset == 'safe':
                            return str(official_safe)
                        elif dataset == 'commonsense':
                            return str(official_commonsense)
                        else:
                            return str(official_gsm8k)
                    else:
                        logger.warning(f"检测到备份数据路径被传入（允许使用）：{dp_path}")

                if dp_path.exists():
                    return str(dp_path)
                else:
                    logger.warning(f"传入的数据路径不存在，将改用正式数据集: {dp_path}")
        except Exception:

            pass

        if dataset == 'safe':
            return str(official_safe)
        elif dataset == 'commonsense':
            return str(official_commonsense)
        return str(official_gsm8k)

    def _get_data_kwargs(self) -> Dict[str, Any]:
        return { 'dataset_type': getattr(self.config, 'dataset_type', 'gsm8k') }

    async def _create_visualizer(self) -> Optional[IVisualizer]:
        if not getattr(self.config, 'visualize', True):
            return None
        try:
            output_dir = getattr(self.config, 'visualizations_dir', getattr(self.config, 'output_dir', 'results'))
            return UnifiedVisualizerAdapter(
                output_dir,
                str(getattr(self.config, 'dataset_type', 'gsm8k')),
                str(getattr(self.config, 'agent_type', 'decoder')),
            )
        except Exception as e:
            logger.warning(f"创建可视化器失败: {e}")
            return None

def create_decoder_probe_runner(config: DecoderProbeConfig) -> DecoderProbeRunner:
    return DecoderProbeRunner(config)

async def run_decoder_probe_experiment(config: DecoderProbeConfig) -> Any:
    runner = create_decoder_probe_runner(config)
    try:
        return await runner.run_experiment()
    finally:
        runner.cleanup_experiment()

