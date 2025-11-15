
import argparse
import os
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from pathlib import Path

try:
    from .global_params import GlobalParams
except ImportError:

    GlobalParams = None

@dataclass
class BaseConfig:

    topology: str = "complete"
    agents: int = 5
    malicious: int = 1
    agent_type: str = "llm"

    strong_model: Optional[str] = None
    weak_model: Optional[str] = None

    dataset_type: str = "gsm8k"                    
    data_path: Optional[str] = None                 

    mode: str = "all"                    
    rounds: int = 1
    question: Optional[str] = None

    position_strategy: str = "random"
    specific_positions: Optional[List[int]] = None
    position_seed: Optional[int] = None
    force_random: bool = False

    save: bool = True
    visualize: bool = True
    output_name: Optional[str] = None
    save_detailed_data: bool = False

    api_key: Optional[str] = None                 
    api_base_url: str = "https://api.openai.com/v1"             

    @staticmethod
    def get_model_path(model_name: str) -> str:

        if GlobalParams is None:
            raise ImportError("GlobalParams not available")

        if model_name.lower() == "llama3":
            return GlobalParams.LLAMA3_MODEL_PATH
        elif model_name.lower() == "llama31":
            return GlobalParams.LLAMA31_MODEL_PATH
        else:
            raise ValueError(f"Unknown model: {model_name}")

SUPPORTED_TOPOLOGIES = [
    "complete", "star", "chain", "tree", "random", "dynamic", "layered_graph"
]

POSITION_STRATEGIES = [
    "random", "star_center", "star_leaf", 
    "tree_root", "tree_internal", "tree_leaf",
    "chain_head", "chain_middle", "chain_tail",
    "layered_top", "layered_middle", "layered_bottom",
    "high_centrality", "low_centrality", "high_degree", "low_degree"
]

AGENT_TYPES = ["traditional", "llm"]

TEST_MODES = ["single", "all"]

class BaseConfigManager:

    def __init__(self, config_class=BaseConfig):
        self.config_class = config_class
        self._config = None

    def create_base_parser(self, description: str = "拜占庭容错系统") -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(
            description=description,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )

        parser.add_argument("--dataset-type", choices=["gsm8k", "safe", "commonsense"],
                           default="gsm8k", help="数据集类型")
        parser.add_argument("--data-path", default=None,
                           help="自定义数据文件路径（可选，优先于默认路径）")

        parser.add_argument("--topology", choices=SUPPORTED_TOPOLOGIES,
                           default="complete", help="网络拓扑类型")
        parser.add_argument("--agents", type=int, default=5, help="智能体总数")
        parser.add_argument("--malicious", type=int, default=1, help="恶意智能体数量")
        parser.add_argument("--agent-type", choices=AGENT_TYPES,
                           default="llm", help="智能体类型")

        parser.add_argument("--strong-model",
                           default=None,
                           help="强模型名称（正常/非恶意智能体使用）")
        parser.add_argument("--weak-model",
                           default=None,
                           help="弱模型名称（恶意/弱智能体使用）")

        parser.add_argument("--mode", choices=TEST_MODES, required=True,
                           help="测试模式: 'single' 测试1个问题, 'all' 测试全部问题")
        parser.add_argument("--rounds", type=int, required=True,
                           help="每个问题的共识轮数")
        parser.add_argument("--question", help="指定问题ID (仅在single模式下)")

        parser.add_argument("--position-strategy", choices=POSITION_STRATEGIES,
                           default="random", help="恶意节点位置策略")
        parser.add_argument("--specific-positions", type=int, nargs='+',
                           help="指定恶意节点位置 (例如: --specific-positions 0 2)")
        parser.add_argument("--position-seed", type=int,
                           help="位置随机种子")
        parser.add_argument("--force-random", action="store_true",
                           help="强制使用随机位置")

        parser.add_argument("--save", action="store_true", default=True,
                           help="保存实验结果")
        parser.add_argument("--no-save", dest="save", action="store_false",
                           help="不保存实验结果")
        parser.add_argument("--visualize", action="store_true", default=True,
                           help="生成可视化图表")
        parser.add_argument("--no-visualize", dest="visualize", action="store_false",
                           help="不生成可视化图表")
        parser.add_argument("--output-name", help="自定义输出文件名")
        parser.add_argument("--save-detailed-data", action="store_true",
                           help="保存详细过程数据")

        parser.add_argument("--llama3-model-path",
                           default="models/LLama-3-8B-Instruct",
                           help="LLaMA3本地模型路径")
        parser.add_argument("--llama31-model-path",
                           default="models/LLama-3.1-8B-Instruct",
                           help="LLaMA3.1本地模型路径")

        parser.add_argument("--api-key", help="API密钥")
        parser.add_argument("--api-base-url", 
                           default="https://api.openai.com/v1",
                           help="API基础URL（OpenAI兼容接口，例如官方 OpenAI 或兼容代理网关）")
        parser.add_argument("--temperature", type=float, default=None,
                           help="生成温度（由方法/数据集决定默认值）")
        parser.add_argument("--max-tokens", type=int, default=1000,
                           help="最大生成token数")
        parser.add_argument("--seed", type=int, default=1234,
                           help="随机种子")

        return parser

    def parse_args_to_config(self, args: argparse.Namespace) -> BaseConfig:    

        api_key = (
            args.api_key
            or os.getenv("OPENAI_COMPATIBILITY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
            or os.getenv("API_KEY")
        )

        api_base_url = args.api_base_url or os.getenv("API_BASE_URL") or "https://api.openai.com/v1"

        return self.config_class(
            dataset_type=getattr(args, 'dataset_type', 'gsm8k'),
            data_path=getattr(args, 'data_path', None),
            topology=args.topology,
            agents=args.agents,
            malicious=args.malicious,
            agent_type=args.agent_type,
            strong_model=getattr(args, 'strong_model', None),
            weak_model=getattr(args, 'weak_model', None),
            mode=args.mode,
            rounds=args.rounds,
            question=args.question,
            position_strategy=args.position_strategy,
            specific_positions=args.specific_positions,
            position_seed=args.position_seed,
            force_random=args.force_random,
            save=args.save,
            visualize=args.visualize,
            output_name=args.output_name,
            save_detailed_data=args.save_detailed_data,
            api_key=api_key,
            api_base_url=api_base_url
        )

    def validate_config(self, config: BaseConfig) -> List[str]:

        return self.validate_base_config(config)

    def validate_base_config(self, config: BaseConfig) -> List[str]:

        errors = []

        if config.agents < 2:
            errors.append("智能体数量必须至少为2")

        if config.malicious >= config.agents:
            errors.append("恶意智能体数量不能大于等于总数量")

        if config.rounds < 1:
            errors.append("共识轮数必须至少为1")

        malicious_ratio = config.malicious / config.agents
        if malicious_ratio > 0.33:
            print(f"警告：恶意节点比例({malicious_ratio:.1%})超过理论容错极限(33%)，实验结果可能不可容错。")

        if config.specific_positions:
            if len(config.specific_positions) != config.malicious:
                errors.append(f"指定位置数量({len(config.specific_positions)})与恶意智能体数量({config.malicious})不匹配")

            if any(pos >= config.agents for pos in config.specific_positions):
                errors.append(f"指定位置超出智能体数量范围(0-{config.agents-1})")

            if len(set(config.specific_positions)) != len(config.specific_positions):
                errors.append("指定位置不能重复")

        paths_to_check = [
            (config.llama3_model_path, "LLaMA3模型路径"),
            (config.llama31_model_path, "LLaMA3.1模型路径"),
        ]

        for path, name in paths_to_check:
            if not Path(path).exists():
                errors.append(f"{name}不存在: {path}")

        return errors

    def print_base_config_summary(self, config: BaseConfig) -> None:

        print("拜占庭容错系统配置:")
        print(f"   拓扑结构: {config.topology}")
        print(f"   智能体总数: {config.agents}")
        print(f"   恶意智能体: {config.malicious} ({config.malicious/config.agents:.1%})")
        print(f"   智能体类型: {config.agent_type}")
        print(f"   测试模式: {config.mode}")
        print(f"   共识轮数: {config.rounds}")
        print(f"   位置策略: {config.position_strategy}")

        if config.specific_positions:
            print(f"   指定位置: {config.specific_positions}")
        if config.position_seed:
            print(f"   位置种子: {config.position_seed}")

        print(f"   保存结果: {'是' if config.save else '否'}")
        print(f"   生成可视化: {'是' if config.visualize else '否'}")
        print(f"   API基础URL: {config.api_base_url}")
        print(f"   温度参数: {config.temperature}")

def create_base_config_from_args(args_list: Optional[List[str]] = None, description: str = "拜占庭容错系统") -> BaseConfig:
    manager = BaseConfigManager()
    parser = manager.create_base_parser(description)
    args = parser.parse_args(args_list)
    return manager.parse_args_to_config(args)

def validate_and_print_config(config: BaseConfig, config_name: str = "系统") -> bool:
    manager = BaseConfigManager()
    errors = manager.validate_base_config(config)

    if errors:
        print(f"{config_name}配置验证失败:")
        for error in errors:
            print(f"   错误: {error}")
        return False

    print(f"{config_name}配置验证通过")
    manager.print_base_config_summary(config)
    return True

if __name__ == "__main__":

    print("测试基础配置系统")
    config = create_base_config_from_args("基础配置测试")

    if validate_and_print_config(config, "基础"):
        print("\n配置系统工作正常")
    else:
        print("\n配置系统存在问题")
