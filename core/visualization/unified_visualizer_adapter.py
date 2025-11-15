#!/usr/bin/env python3

from __future__ import annotations

import os
import json
import logging
from typing import Any, Optional

from .core_metrics_visualizer import create_core_metrics_visualization
from .attack_effect_visualizer import create_attack_effect_analysis
from .comprehensive_analyzer import ComprehensiveAnalyzer

logger = logging.getLogger(__name__)

class UnifiedVisualizerAdapter:
    def __init__(self, output_dir: str, dataset_type: str, agent_type: str):
        self.output_dir = output_dir
        self.dataset_type = dataset_type
        self.agent_type = agent_type
        self.comprehensive_analyzer = ComprehensiveAnalyzer()

    def _load_saved_json(self, output_dir: str) -> Optional[str]:
        try:
            files = [f for f in os.listdir(output_dir) if f.endswith('.json') and not f.endswith('.summary.json')]
            if not files:
                return None

            json_path = os.path.join(output_dir, files[0])
            if os.path.exists(json_path):
                return json_path
            return None
        except Exception as e:
            logger.warning(f"查找JSON文件失败: {e}")
            return None

    def generate_comprehensive_report(self, experiment_result: Any, output_dir: str) -> None:
        try:
            os.makedirs(output_dir, exist_ok=True)

            json_path = self._load_saved_json(output_dir)
            if not json_path:
                logger.warning("未找到已保存的JSON文件，跳过可视化生成（建议先确保结果保存成功）")
                return

            with open(json_path, 'r') as f:
                saved_data = json.load(f)

            generated_files = []

            try:
                core_metrics_dir = create_core_metrics_visualization(saved_data, output_dir)
                if core_metrics_dir:
                    generated_files.append(core_metrics_dir)
            except Exception as e:
                logger.warning(f"核心指标可视化生成失败: {e}")

            try:
                attack_file = create_attack_effect_analysis(saved_data, output_dir)
                if attack_file:
                    generated_files.append(attack_file)
            except Exception as e:
                logger.warning(f"攻击效果分析可视化生成失败: {e}")

            try:

                base = os.path.basename(json_path)
                report_filename = base.replace('.json', '_report.txt')
                report_path = os.path.join(output_dir, report_filename)
                analysis_file = self.comprehensive_analyzer.analyze_in_memory(saved_data, report_path)
                if analysis_file:
                    generated_files.append(analysis_file)
            except Exception as e:
                logger.warning(f"综合分析生成失败: {e}")

            logger.info(f"可视化生成完成，共 {len(generated_files)} 个输出")
        except Exception as e:
            logger.error(f"统一可视化适配器运行失败: {e}")

