#!/usr/bin/env python3

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Union
import pandas as pd

from ..interfaces import ExperimentResult, MethodType
from .result_processor import StandardizedResultProcessor

logger = logging.getLogger(__name__)

class UnifiedResultManager:

    def __init__(self, base_results_dir: str = "results"):
        self.base_results_dir = Path(base_results_dir)
        self.base_results_dir.mkdir(parents=True, exist_ok=True)

        self._create_directory_structure()

        self.result_processor = StandardizedResultProcessor(str(self.base_results_dir))

        logger.info(f"统一结果管理器初始化完成: {self.base_results_dir}")

    def _create_directory_structure(self) -> None:

        for method in MethodType:
            method_dir = self.base_results_dir / method.value
            method_dir.mkdir(exist_ok=True)

            (method_dir / "experiments").mkdir(exist_ok=True)
            (method_dir / "analysis").mkdir(exist_ok=True)
            (method_dir / "exports").mkdir(exist_ok=True)
            (method_dir / "visualizations").mkdir(exist_ok=True)

        (self.base_results_dir / "comparative").mkdir(exist_ok=True)
        (self.base_results_dir / "archives").mkdir(exist_ok=True)
        (self.base_results_dir / "temp").mkdir(exist_ok=True)

    def save_experiment_result(
        self, 
        result: ExperimentResult, 
        experiment_name: Optional[str] = None
    ) -> str:

        method_dir = self.base_results_dir / result.method_type.value / "experiments"

        if experiment_name:
            filename = f"{experiment_name}_{result.experiment_id}.json"
        else:
            filename = f"{result.experiment_id}.json"

        result_path = method_dir / filename

        self.result_processor.save_experiment_result(result, str(result_path))

        self._update_experiment_index(result, str(result_path))

        logger.info(f"实验结果已保存: {result_path}")
        return str(result_path)

    def load_experiment_result(self, result_path: str) -> ExperimentResult:

        return self.result_processor.load_experiment_result(result_path)

    def list_experiments(
        self, 
        method_type: Optional[MethodType] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        limit: Optional[int] = None
    ) -> List[Dict[str, Any]]:

        experiments = []

        if method_type:
            search_dirs = [self.base_results_dir / method_type.value / "experiments"]
        else:
            search_dirs = [
                self.base_results_dir / method.value / "experiments" 
                for method in MethodType
            ]

        for search_dir in search_dirs:
            if not search_dir.exists():
                continue

            for result_file in search_dir.glob("*.json"):
                if result_file.name.endswith(".summary.json"):
                    continue

                try:

                    summary_file = result_file.with_suffix(".summary.json")
                    if summary_file.exists():
                        with open(summary_file, 'r', encoding='utf-8') as f:
                            summary = json.load(f)
                    else:

                        result = self.load_experiment_result(str(result_file))
                        summary = self._create_experiment_summary(result)

                    result_time = datetime.fromisoformat(summary.get("timestamp", "1970-01-01"))
                    if start_date and result_time < start_date:
                        continue
                    if end_date and result_time > end_date:
                        continue

                    summary["file_path"] = str(result_file)
                    experiments.append(summary)

                except Exception as e:
                    logger.warning(f"读取实验结果失败: {result_file}, {e}")
                    continue

        experiments.sort(key=lambda x: x.get("timestamp", ""), reverse=True)

        if limit:
            experiments = experiments[:limit]

        return experiments

    def search_experiments(
        self,
        query: Dict[str, Any],
        method_type: Optional[MethodType] = None
    ) -> List[Dict[str, Any]]:

        experiments = self.list_experiments(method_type)

        filtered_experiments = []

        for exp in experiments:
            match = True

            for key, value in query.items():
                if key not in exp:
                    match = False
                    break

                exp_value = exp[key]

                if isinstance(value, str):
                    if str(exp_value).lower().find(value.lower()) == -1:
                        match = False
                        break
                elif isinstance(value, (int, float)):
                    if exp_value != value:
                        match = False
                        break
                elif isinstance(value, dict) and "min" in value:
                    if exp_value < value["min"]:
                        match = False
                        break
                elif isinstance(value, dict) and "max" in value:
                    if exp_value > value["max"]:
                        match = False
                        break

            if match:
                filtered_experiments.append(exp)

        return filtered_experiments

    def compare_experiments(
        self, 
        experiment_ids: List[str],
        comparison_name: Optional[str] = None
    ) -> Dict[str, Any]:

        if len(experiment_ids) < 2:
            raise ValueError("至少需要2个实验进行比较")

        experiments = []
        for exp_id in experiment_ids:
            exp_file = self._find_experiment_file(exp_id)
            if exp_file:
                result = self.load_experiment_result(exp_file)
                experiments.append(result)
            else:
                logger.warning(f"未找到实验: {exp_id}")

        if len(experiments) < 2:
            raise ValueError("找到的有效实验少于2个")

        comparison = self._create_comparison_analysis(experiments)

        if comparison_name:
            comparison_file = self.base_results_dir / "comparative" / f"{comparison_name}.json"
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            comparison_file = self.base_results_dir / "comparative" / f"comparison_{timestamp}.json"

        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)

        logger.info(f"实验比较结果已保存: {comparison_file}")
        return comparison

    def export_results(
        self,
        experiment_ids: List[str],
        export_format: str = "excel",
        output_path: Optional[str] = None
    ) -> str:

        if not experiment_ids:
            raise ValueError("未指定要导出的实验")

        if output_path is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_path = str(self.base_results_dir / "exports" / f"export_{timestamp}.{export_format}")

        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        experiments = []
        for exp_id in experiment_ids:
            exp_file = self._find_experiment_file(exp_id)
            if exp_file:
                result = self.load_experiment_result(exp_file)
                experiments.append(result)

        if not experiments:
            raise ValueError("未找到有效的实验结果")

        if export_format.lower() in ["excel", "xlsx"]:
            self._export_to_excel(experiments, output_file)
        elif export_format.lower() == "csv":
            self._export_to_csv(experiments, output_file)
        elif export_format.lower() == "json":
            self._export_to_json(experiments, output_file)
        else:
            raise ValueError(f"不支持的导出格式: {export_format}")

        logger.info(f"结果导出完成: {output_file}")
        return str(output_file)

    def archive_experiments(
        self,
        experiment_ids: List[str],
        archive_name: Optional[str] = None
    ) -> str:

        if not experiment_ids:
            raise ValueError("未指定要归档的实验")

        if archive_name:
            archive_dir = self.base_results_dir / "archives" / archive_name
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            archive_dir = self.base_results_dir / "archives" / f"archive_{timestamp}"

        archive_dir.mkdir(parents=True, exist_ok=True)

        archived_files = []
        for exp_id in experiment_ids:
            exp_file = self._find_experiment_file(exp_id)
            if exp_file:

                dest_file = archive_dir / Path(exp_file).name
                shutil.copy2(exp_file, dest_file)
                archived_files.append(str(dest_file))

                summary_file = Path(exp_file).with_suffix(".summary.json")
                if summary_file.exists():
                    dest_summary = archive_dir / summary_file.name
                    shutil.copy2(summary_file, dest_summary)

        archive_index = {
            "archive_name": archive_name or f"archive_{timestamp}",
            "created_at": datetime.now().isoformat(),
            "experiment_count": len(archived_files),
            "experiments": experiment_ids,
            "files": archived_files
        }

        index_file = archive_dir / "archive_index.json"
        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(archive_index, f, indent=2, ensure_ascii=False)

        logger.info(f"实验归档完成: {archive_dir}")
        return str(archive_dir)

    def cleanup_temp_files(self, older_than_days: int = 7) -> int:

        temp_dir = self.base_results_dir / "temp"

        if not temp_dir.exists():
            return 0

        cutoff_time = datetime.now().timestamp() - (older_than_days * 24 * 3600)
        cleaned_count = 0

        for temp_file in temp_dir.rglob("*"):
            if temp_file.is_file() and temp_file.stat().st_mtime < cutoff_time:
                try:
                    temp_file.unlink()
                    cleaned_count += 1
                except Exception as e:
                    logger.warning(f"清理临时文件失败: {temp_file}, {e}")

        logger.info(f"清理了 {cleaned_count} 个临时文件")
        return cleaned_count

    def get_statistics(self) -> Dict[str, Any]:

        stats = {
            "total_experiments": 0,
            "by_method": {},
            "by_date": {},
            "storage_size": 0
        }

        for method in MethodType:
            method_dir = self.base_results_dir / method.value / "experiments"
            if method_dir.exists():
                count = len(list(method_dir.glob("*.json")))

                count = count - len(list(method_dir.glob("*.summary.json")))
                stats["by_method"][method.value] = count
                stats["total_experiments"] += count

        stats["storage_size"] = sum(
            f.stat().st_size for f in self.base_results_dir.rglob("*") if f.is_file()
        )

        return stats

    def _update_experiment_index(self, result: ExperimentResult, file_path: str) -> None:

        index_file = self.base_results_dir / "experiment_index.json"

        if index_file.exists():
            with open(index_file, 'r', encoding='utf-8') as f:
                index = json.load(f)
        else:
            index = {"experiments": []}

        entry = {
            "experiment_id": result.experiment_id,
            "method_type": result.method_type.value,
            "timestamp": datetime.now().isoformat(),
            "file_path": file_path,
            "agent_count": result.agent_count,
            "malicious_count": result.malicious_count,
            "execution_time": result.execution_time
        }

        index["experiments"].append(entry)

        if len(index["experiments"]) > 1000:
            index["experiments"] = index["experiments"][-1000:]

        with open(index_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2, ensure_ascii=False)

    def _find_experiment_file(self, experiment_id: str) -> Optional[str]:

        for method in MethodType:
            method_dir = self.base_results_dir / method.value / "experiments"
            if not method_dir.exists():
                continue

            for result_file in method_dir.glob(f"*{experiment_id}*.json"):
                if not result_file.name.endswith(".summary.json"):
                    return str(result_file)

        return None

    def _create_experiment_summary(self, result: ExperimentResult) -> Dict[str, Any]:

        return {
            "experiment_id": result.experiment_id,
            "method_type": result.method_type.value,
            "topology_type": result.topology_type.value if hasattr(result.topology_type, 'value') else str(result.topology_type),
            "agent_count": result.agent_count,
            "malicious_count": result.malicious_count,
            "execution_time": result.execution_time,
            "timestamp": datetime.now().isoformat(),
            "question_count": len(result.questions),
            "consensus_count": len(result.consensus_results)
        }

    def _create_comparison_analysis(self, experiments: List[ExperimentResult]) -> Dict[str, Any]:

        comparison = {
            "comparison_id": f"comp_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "timestamp": datetime.now().isoformat(),
            "experiment_count": len(experiments),
            "experiments": [],
            "comparative_analysis": {}
        }

        for exp in experiments:
            exp_info = {
                "experiment_id": exp.experiment_id,
                "method_type": exp.method_type.value,
                "agent_count": exp.agent_count,
                "malicious_count": exp.malicious_count,
                "execution_time": exp.execution_time
            }
            comparison["experiments"].append(exp_info)

        if experiments:
            comparison["comparative_analysis"] = {
                "avg_execution_time": sum(e.execution_time for e in experiments) / len(experiments),
                "min_execution_time": min(e.execution_time for e in experiments),
                "max_execution_time": max(e.execution_time for e in experiments),
                "avg_agent_count": sum(e.agent_count for e in experiments) / len(experiments),
                "method_distribution": {}
            }

            method_counts = {}
            for exp in experiments:
                method = exp.method_type.value
                method_counts[method] = method_counts.get(method, 0) + 1
            comparison["comparative_analysis"]["method_distribution"] = method_counts

        return comparison

    def _export_to_excel(self, experiments: List[ExperimentResult], output_file: Path) -> None:

        try:
            with pd.ExcelWriter(output_file.with_suffix('.xlsx'), engine='openpyxl') as writer:

                summary_data = []
                for exp in experiments:
                    summary_data.append({
                        "实验ID": exp.experiment_id,
                        "方法类型": exp.method_type.value,
                        "智能体数量": exp.agent_count,
                        "恶意智能体数量": exp.malicious_count,
                        "执行时间": exp.execution_time,
                        "问题数量": len(exp.questions),
                        "共识结果数量": len(exp.consensus_results)
                    })

                summary_df = pd.DataFrame(summary_data)
                summary_df.to_excel(writer, sheet_name='实验摘要', index=False)

                if len(experiments) <= 5:
                    for i, exp in enumerate(experiments):
                        sheet_name = f"实验{i+1}_{exp.experiment_id[:8]}"

                        consensus_data = []
                        for cr in exp.consensus_results:
                            consensus_data.append({
                                "问题ID": cr.question_id,
                                "共识答案": cr.consensus_answer,
                                "置信度": cr.consensus_confidence,
                                "参与者数量": cr.participant_count,
                                "轮次": cr.round_number,
                                "是否收敛": cr.convergence_achieved
                            })

                        if consensus_data:
                            consensus_df = pd.DataFrame(consensus_data)
                            consensus_df.to_excel(writer, sheet_name=sheet_name, index=False)

        except ImportError:
            logger.warning("pandas或openpyxl未安装，回退到JSON格式")
            self._export_to_json(experiments, output_file.with_suffix('.json'))

    def _export_to_csv(self, experiments: List[ExperimentResult], output_file: Path) -> None:

        summary_data = []
        for exp in experiments:
            summary_data.append({
                "experiment_id": exp.experiment_id,
                "method_type": exp.method_type.value,
                "agent_count": exp.agent_count,
                "malicious_count": exp.malicious_count,
                "execution_time": exp.execution_time,
                "question_count": len(exp.questions),
                "consensus_count": len(exp.consensus_results)
            })

        try:
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(output_file.with_suffix('.csv'), index=False, encoding='utf-8')
        except ImportError:

            import csv
            with open(output_file.with_suffix('.csv'), 'w', newline='', encoding='utf-8') as f:
                if summary_data:
                    writer = csv.DictWriter(f, fieldnames=summary_data[0].keys())
                    writer.writeheader()
                    writer.writerows(summary_data)

    def _export_to_json(self, experiments: List[ExperimentResult], output_file: Path) -> None:

        export_data = {
            "export_timestamp": datetime.now().isoformat(),
            "experiment_count": len(experiments),
            "experiments": []
        }

        for exp in experiments:
            exp_data = self.result_processor._make_serializable(exp)
            export_data["experiments"].append(exp_data)

        with open(output_file.with_suffix('.json'), 'w', encoding='utf-8') as f:
            json.dump(export_data, f, indent=2, ensure_ascii=False)

_manager_instance = None

def get_result_manager(base_results_dir: str = "results") -> UnifiedResultManager:

    global _manager_instance
    if _manager_instance is None:
        _manager_instance = UnifiedResultManager(base_results_dir)
    return _manager_instance

def save_result(result: ExperimentResult, experiment_name: Optional[str] = None) -> str:

    manager = get_result_manager()
    return manager.save_experiment_result(result, experiment_name)

def load_result(result_path: str) -> ExperimentResult:

    manager = get_result_manager()
    return manager.load_experiment_result(result_path)

def list_results(method_type: Optional[MethodType] = None, limit: Optional[int] = None) -> List[Dict[str, Any]]:

    manager = get_result_manager()
    return manager.list_experiments(method_type, limit=limit)

if __name__ == "__main__":

    print("统一结果管理器测试")

    manager = get_result_manager("test_results")
    print(f"管理器初始化完成: {manager.base_results_dir}")

    stats = manager.get_statistics()
    print(f"统计信息: {stats}")

    print("统一结果管理器测试完成")
