#!/usr/bin/env python3

import asyncio
import logging
import sys
import os
from pathlib import Path
from typing import Any, Optional

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from config.unified_config_manager import UnifiedConfigManager, MethodType as ConfigMethodType
from config.prompt_probe_config import create_prompt_probe_config
from config.decoder_probe_config import create_decoder_probe_config
from config.base_config import BaseConfig
from core.interfaces import MethodType

logger = logging.getLogger(__name__)

def _normalize_proxy_env() -> None:

    try:
        pairs = [("HTTP_PROXY", "http_proxy"), ("HTTPS_PROXY", "https_proxy")]
        for upper_key, lower_key in pairs:
            upper_val = os.environ.get(upper_key)
            lower_val = os.environ.get(lower_key)
            if upper_val and not lower_val:
                os.environ[lower_key] = upper_val
            if lower_val and not upper_val:
                os.environ[upper_key] = lower_val
    except Exception:

        pass

def _load_env_files() -> None:

    try:
        from dotenv import load_dotenv  # type: ignore
    except Exception:

        return

    env_candidates = [project_root / ".env", Path.cwd() / ".env"]
    for env_path in env_candidates:
        try:
            if env_path.exists():
                load_dotenv(dotenv_path=str(env_path), override=False)
        except Exception:

            continue

    _normalize_proxy_env()

_load_env_files()

# ---------------------------------------------------------------------------
# i18n – import lazily so the module works even if core/ is not yet on path
# ---------------------------------------------------------------------------
try:
    from core.i18n import set_language, install_translating_formatter as _install_fmt
except ImportError:
    set_language = lambda lang: None  # noqa: E731
    _install_fmt = lambda **kw: None  # noqa: E731


class UnifiedMethodEntry:

    def __init__(self):
        self.config_manager = UnifiedConfigManager()
        logger.info("初始化统一方法入口点")

    async def run_experiment(self, method: str, config_path: Optional[str] = None, extra_args: Optional[list] = None, **kwargs) -> Any:

        method_type = self._parse_method_type(method)

        logger.info(f"开始运行{method_type.value}实验...")

        try:

            if config_path:
                config = self._load_config_from_file(method_type, config_path)
            else:
                config = self._parse_command_line_config(method_type, extra_args=extra_args or [], **kwargs)

            result = await self._route_to_runner(method_type, config)

            logger.info(f"{method_type.value}实验完成: {result.experiment_id}")
            return result

        except Exception as e:
            logger.error(f"{method_type.value}实验失败: {e}")
            raise

    def _parse_method_type(self, method: str) -> MethodType:

        method_lower = method.lower().replace('_', '').replace('-', '')

        if method_lower in ['pilot']:
            return MethodType.PILOT
        elif method_lower in ['promptprobe', 'probe', 'prompt']:
            return MethodType.PROMPT_PROBE
        elif method_lower in ['safe', 'safedecoder']:
            raise NotImplementedError("safe 方法已禁用")
        elif method_lower in ['decoderprobe', 'decoder', 'hcp']:
            return MethodType.DECODER
        else:
            raise ValueError(f"不支持的方法类型: {method}")

    def _load_config_from_file(self, method_type: MethodType, config_path: str) -> Any:

        logger.debug(f"从文件加载{method_type.value}配置: {config_path}")

        def _to_config_method_type(mt: MethodType) -> ConfigMethodType:
            return ConfigMethodType(mt.value)

        config = self.config_manager.load_config(
            _to_config_method_type(method_type), config_file=config_path
        )

        if not config:
            raise ValueError(f"无法加载{method_type.value}配置文件: {config_path}")

        return config

    def _parse_command_line_config(self, method_type: MethodType, extra_args: Optional[list] = None, **kwargs) -> Any:

        logger.debug(f"解析{method_type.value}命令行配置")

        if method_type == MethodType.PILOT:

            from config.base_config import create_base_config_from_args
            return create_base_config_from_args(args_list=extra_args or [])
        elif method_type == MethodType.PROMPT_PROBE:
            return create_prompt_probe_config(extra_args)
        elif method_type == MethodType.DECODER:
            return create_decoder_probe_config(extra_args)
        else:
            raise ValueError(f"不支持的方法类型: {method_type}")

    async def _route_to_runner(self, method_type: MethodType, config: Any) -> Any:

        logger.debug(f"路由到{method_type.value}运行器")

        if method_type == MethodType.PILOT:

            from core.runners.pilot_runner import run_pilot_experiment as run_gsm8k_experiment
            return await run_gsm8k_experiment(config)
        elif method_type == MethodType.PROMPT_PROBE:

            from core.runners.prompt_probe_runner import run_prompt_probe_experiment
            return await run_prompt_probe_experiment(config)
        elif method_type == MethodType.DECODER:
            from core.runners.decoder_probe_runner import run_decoder_probe_experiment
            return await run_decoder_probe_experiment(config)
        else:
            raise ValueError(f"不支持的方法类型: {method_type}")

    def list_supported_methods(self) -> list:

        return [
            {
                "name": "pilot",
                "aliases": [],
                "description": "先导试验（Pilot）统一方法入口"
            },
            {
                "name": "prompt_probe",
                "aliases": ["prompt-probe", "probe"],
                "description": "Prompt Probe置信度探测方法"
            },

            {
                "name": "decoder_probe",
                "aliases": ["decoder", "hcp"],
                "description": "本地解码器隐层置信度探针（HCP），支持GSM8K/SAFE"
            }
        ]

    def get_method_help(self, method: str) -> str:

        try:
            method_type = self._parse_method_type(method)

            if method_type == MethodType.PILOT:

                from config.base_config import BaseConfigManager
                parser = BaseConfigManager().create_base_parser(description="Pilot 先导试验方法")
                return parser.format_help()
            elif method_type == MethodType.PROMPT_PROBE:

                from config.prompt_probe_config import PromptProbeConfigManager
                parser = PromptProbeConfigManager().create_parser()
                return parser.format_help()
            elif method_type == MethodType.DECODER:

                from config import decoder_probe_config as dpc
                parser = dpc._create_parser()  # type: ignore[attr-defined]
                return parser.format_help()
            else:
                return f"未知方法: {method}"
        except Exception as e:
            return f"获取帮助失败: {e}"

_entry_instance = None

def get_unified_entry() -> UnifiedMethodEntry:

    global _entry_instance
    if _entry_instance is None:
        _entry_instance = UnifiedMethodEntry()
    return _entry_instance

async def run_method_experiment(method: str, config_path: Optional[str] = None, **kwargs) -> Any:

    entry = get_unified_entry()
    return await entry.run_experiment(method, config_path, **kwargs)

def main():

    import argparse

    if "--list-methods" in sys.argv:
        entry = get_unified_entry()
        methods = entry.list_supported_methods()
        print("支持的方法:")
        for method in methods:
            print(f"  {method['name']}: {method['description']}")
            if method['aliases']:
                print(f"    别名: {', '.join(method['aliases'])}")
        return

    if "--method-help" in sys.argv:
        entry = get_unified_entry()
        try:
            idx = sys.argv.index("--method-help")
            if idx + 1 < len(sys.argv):
                method_name = sys.argv[idx + 1]
                help_text = entry.get_method_help(method_name)
                print(help_text)
            else:
                print("错误: --method-help 需要指定方法名称")
                sys.exit(1)
        except Exception as e:
            print(f"错误: {e}")
            sys.exit(1)
        return

    if len(sys.argv) >= 2 and sys.argv[1] in ["pilot", "prompt_probe", "decoder_probe", "decoder"] and "--help" in sys.argv:
        entry = get_unified_entry()
        help_text = entry.get_method_help(sys.argv[1])
        print(help_text)
        return

    parser = argparse.ArgumentParser(
        description="Byzantine项目统一方法入口点",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        add_help=False
    )

    parser.add_argument(
        "method",
        choices=["pilot", "prompt_probe", "decoder_probe", "decoder"],
        help="要运行的方法 (pilot: 先导试验, prompt_probe: 提示探针, decoder_probe/decoder: 解码器探针)"
    )

    parser.add_argument(
        "--config",
        type=str,
        help="配置文件路径"
    )

    parser.add_argument(
        "--log-level",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        default="INFO",
        help="Log level"
    )

    parser.add_argument(
        "--lang",
        choices=["en", "zh"],
        default="zh",
        help="Output language for log messages: 'en' for English, 'zh' for Chinese (default)"
    )

    parser.add_argument(
        "-h", "--help",
        action="store_true",
        help="Show help"
    )

    args, unknown_args = parser.parse_known_args()

    if args.help:
        if args.method:
            entry = get_unified_entry()
            help_text = entry.get_method_help(args.method)
            print(help_text)
        else:
            parser.print_help()
        return

    # Configure language before any logging output is produced
    set_language(args.lang)

    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    if args.lang == "en":
        _install_fmt(fmt='%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    entry = get_unified_entry()

    async def run_experiment():
        try:
            result = await entry.run_experiment(args.method, args.config, extra_args=unknown_args)

            if args.lang == "en":
                print(f"\nExperiment complete!")
                print(f"Experiment ID: {result.experiment_id}")
                print(f"Method type: {result.method_type.value}")
                print(f"Agent count: {result.agent_count}")
                print(f"Malicious agent count: {result.malicious_count}")
                print(f"Execution time: {result.execution_time:.2f}s")

                if result.evaluation_metrics:
                    overall = result.evaluation_metrics.get('overall_assessment', {})
                    if 'accuracy' in overall:
                        print(f"Accuracy: {overall['accuracy']:.4f}")
                    if 'consensus_rate' in overall:
                        print(f"Consensus rate: {overall['consensus_rate']:.4f}")

                    safety = result.evaluation_metrics.get('safety_assessment', {})
                    if safety:
                        print(f"Safety rate: {safety.get('safety_rate', 'N/A')}")
            else:
                print(f"\n实验完成!")
                print(f"实验ID: {result.experiment_id}")
                print(f"方法类型: {result.method_type.value}")
                print(f"智能体数量: {result.agent_count}")
                print(f"恶意智能体数量: {result.malicious_count}")
                print(f"执行时间: {result.execution_time:.2f}秒")

                if result.evaluation_metrics:
                    overall = result.evaluation_metrics.get('overall_assessment', {})
                    if 'accuracy' in overall:
                        print(f"准确率: {overall['accuracy']:.4f}")
                    if 'consensus_rate' in overall:
                        print(f"共识率: {overall['consensus_rate']:.4f}")

                    safety = result.evaluation_metrics.get('safety_assessment', {})
                    if safety:
                        print(f"安全率: {safety.get('safety_rate', 'N/A')}")

        except Exception as e:
            logger.error(f"Experiment failed: {e}" if args.lang == "en" else f"实验失败: {e}")
            sys.exit(1)

    asyncio.run(run_experiment())

if __name__ == "__main__":
    main()
