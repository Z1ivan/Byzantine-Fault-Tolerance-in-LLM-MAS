from dataclasses import dataclass
from typing import Optional, List
import argparse

@dataclass
class DecoderProbeConfig:

    topology: str = "complete"
    agents: int = 7
    malicious: int = 6
    agent_type: str = "decoder"
    mode: str = "all"              
    rounds: int = 1
    question: Optional[str] = None
    dataset_type: str = "gsm8k"                          

    data_path: Optional[str] = None

    save: bool = True
    visualize: bool = True
    output_name: Optional[str] = None

    llama3_model_path: str = "models/LLama-3-8B-Instruct"
    llama31_model_path: str = "models/LLama-3.1-8B-Instruct"

    gsm8k_llama3_lcd: str = "lcd_models/gsm8k/3_pooled_layer16_pca256_logistic"
    gsm8k_llama31_lcd: str = "lcd_models/gsm8k/3.1_pooled_layer12_pca256_logistic"

    safe_llama3_lcd: str = "lcd_models/safe/3_pooled_layer17_pca256_logistic"
    safe_llama31_lcd: str = "lcd_models/safe/3.1_pooled_layer12_pca256_logistic"

    commonsense_llama3_lcd: str = "lcd_models/commonsense/3_pooled_layer14_pca256_mlp"
    commonsense_llama31_lcd: str = "lcd_models/commonsense/3.1_query_layer14_pca256_mlp"

    max_new_tokens: int = 512
    do_sample: bool = False
    temperature: float = 0.0
    seed: int = 1234

    position_strategy: str = "random"
    specific_positions: Optional[List[int]] = None
    position_seed: Optional[int] = None
    force_random: bool = False

def _create_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Decoder Probe（HCP）",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument("--dataset-type", choices=["gsm8k", "safe", "commonsense"], default="gsm8k")
    parser.add_argument("--data-path", default=None, help="自定义数据文件路径，优先于默认数据集")
    parser.add_argument("--mode", choices=["single", "all"], default="all")
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--question", help="当 mode=single 时指定问题ID，例如 safe_001 或 gsm8k_001")
    parser.add_argument("--topology", default="complete")
    parser.add_argument("--agents", type=int, default=7)
    parser.add_argument("--malicious", type=int, default=6)
    parser.add_argument("--agent-type", default="decoder")

    parser.add_argument("--save", action="store_true")
    parser.add_argument("--no-save", dest="save", action="store_false")
    parser.set_defaults(save=True)
    parser.add_argument("--visualize", action="store_true")
    parser.add_argument("--no-visualize", dest="visualize", action="store_false")
    parser.set_defaults(visualize=True)
    parser.add_argument("--output-name")

    parser.add_argument("--llama3-model-path", default=DecoderProbeConfig.llama3_model_path)
    parser.add_argument("--llama31-model-path", default=DecoderProbeConfig.llama31_model_path)

    parser.add_argument("--gsm8k-llama3-lcd", default=DecoderProbeConfig.gsm8k_llama3_lcd)
    parser.add_argument("--gsm8k-llama31-lcd", default=DecoderProbeConfig.gsm8k_llama31_lcd)
    parser.add_argument("--safe-llama3-lcd", default=DecoderProbeConfig.safe_llama3_lcd)
    parser.add_argument("--safe-llama31-lcd", default=DecoderProbeConfig.safe_llama31_lcd)
    parser.add_argument("--commonsense-llama3-lcd", default=DecoderProbeConfig.commonsense_llama3_lcd)
    parser.add_argument("--commonsense-llama31-lcd", default=DecoderProbeConfig.commonsense_llama31_lcd)

    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--do-sample", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=1234)

    parser.add_argument("--position-strategy", default="random")
    parser.add_argument("--specific-positions", type=int, nargs='+')
    parser.add_argument("--position-seed", type=int)
    parser.add_argument("--force-random", action="store_true")
    return parser

def create_decoder_probe_config(extra_args: Optional[list]) -> DecoderProbeConfig:
    parser = _create_parser()
    args = parser.parse_args(extra_args)
    return DecoderProbeConfig(
        topology=args.topology,
        agents=args.agents,
        malicious=args.malicious,
        agent_type=args.agent_type,
        mode=args.mode,
        rounds=args.rounds,
        question=args.question,
        dataset_type=args.dataset_type,
        data_path=args.data_path,
        save=args.save,
        visualize=args.visualize,
        output_name=args.output_name,
        llama3_model_path=args.llama3_model_path,
        llama31_model_path=args.llama31_model_path,
        gsm8k_llama3_lcd=args.gsm8k_llama3_lcd,
        gsm8k_llama31_lcd=args.gsm8k_llama31_lcd,
        safe_llama3_lcd=args.safe_llama3_lcd,
        safe_llama31_lcd=args.safe_llama31_lcd,
        commonsense_llama3_lcd=args.commonsense_llama3_lcd,
        commonsense_llama31_lcd=args.commonsense_llama31_lcd,
        max_new_tokens=args.max_new_tokens,
        do_sample=args.do_sample,
        temperature=args.temperature,
        seed=args.seed,
        position_strategy=args.position_strategy,
        specific_positions=args.specific_positions,
        position_seed=args.position_seed,
        force_random=args.force_random,
    )

