from dataclasses import dataclass
from typing import Dict, Any, Optional, Tuple
from pathlib import Path
import json

@dataclass
class LLMGenerationParams:

    max_new_tokens: int
    do_sample: bool
    temperature: float
    repetition_penalty: float
    num_return_sequences: int
    use_cache: bool

    top_p: Optional[float] = None
    top_k: Optional[int] = None

    seed: int = 1234

    dataset_type: str = ""
    method: str = ""                                    
    source: str = ""                
    description: str = ""

    def to_generate_kwargs(self, tokenizer) -> Dict[str, Any]:

        kwargs = {
            'max_new_tokens': self.max_new_tokens,
            'use_cache': self.use_cache,
            'pad_token_id': tokenizer.eos_token_id,
            'eos_token_id': tokenizer.eos_token_id,
        }

        if self.do_sample:

            kwargs['do_sample'] = True
            kwargs['temperature'] = self.temperature
            kwargs['repetition_penalty'] = self.repetition_penalty
            kwargs['num_return_sequences'] = self.num_return_sequences

            if self.top_p is not None:
                kwargs['top_p'] = self.top_p
            if self.top_k is not None:
                kwargs['top_k'] = self.top_k
        else:

            kwargs['do_sample'] = False
            kwargs['repetition_penalty'] = self.repetition_penalty
            kwargs['num_return_sequences'] = self.num_return_sequences

        return kwargs

    def to_dict(self) -> Dict[str, Any]:

        return {
            'max_new_tokens': self.max_new_tokens,
            'do_sample': self.do_sample,
            'temperature': self.temperature,
            'top_p': self.top_p,
            'top_k': self.top_k,
            'repetition_penalty': self.repetition_penalty,
            'num_return_sequences': self.num_return_sequences,
            'use_cache': self.use_cache,
            'seed': self.seed,
            'dataset_type': self.dataset_type,
            'method': self.method,
            'source': self.source,
            'description': self.description,
        }

@dataclass
class LCDProbeConfig:

    model_path: str

    target_layer: int         
    pooling_method: str                          
    pca_dim: Optional[int] = None             

    probe_type: str = "logistic"                    

    dataset_type: str = ""
    model_name: str = ""                  
    source: str = ""
    description: str = ""

    def to_dict(self) -> Dict[str, Any]:

        return {
            'model_path': self.model_path,
            'target_layer': self.target_layer,
            'pooling_method': self.pooling_method,
            'pca_dim': self.pca_dim,
            'probe_type': self.probe_type,
            'dataset_type': self.dataset_type,
            'model_name': self.model_name,
            'source': self.source,
            'description': self.description,
        }

class GlobalParams:

    LLAMA3_MODEL_PATH = "models/LLama-3-8B-Instruct"
    LLAMA31_MODEL_PATH = "models/LLama-3.1-8B-Instruct"

    DECODER_GSM8K_DATA = "data/byzantine/gsm8k/llama3.1_10.json"
    DECODER_SAFE_DATA = "data/byzantine/safe/all_llama31_win.json"
    DECODER_COMMONSENSE_DATA = "data/byzantine/commonsense/llama31_win_10.json"

    PILOT_GSM8K_DATA = "data/byzantine/gsm8k/gsm8k_final_dataset_20250723_154643.json"
    PILOT_SAFE_DATA = "data/byzantine/safe/safe_final_dataset_aligned_20250723_191623.json"

    DECODER_GSM8K_GENERATION = LLMGenerationParams(
        max_new_tokens=256,
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="gsm8k",
        method="decoder_probe",
        source="decoder_gsm8k_training_config",
        description="GSM8K数学推理（采样生成，temperature=0.1）"
    )

    DECODER_SAFE_GENERATION = LLMGenerationParams(
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="safe",
        method="decoder_probe",
        source="decoder_safe_training_config",
        description="Safe安全性（确定性生成，完全可复现）"
    )

    PILOT_GSM8K_GENERATION = LLMGenerationParams(
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="gsm8k",
        method="pilot",
        source="Materials Appendix C",
        description="Pilot实验GSM8K（API调用）"
    )

    PILOT_SAFE_GENERATION = LLMGenerationParams(
        max_new_tokens=1000,
        do_sample=True,
        temperature=0.1,
        repetition_penalty=1.0,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="safe",
        method="pilot",
        source="Materials Appendix C",
        description="Pilot实验Safe（API调用）"
    )

    PROMPT_PROBE_GSM8K_GENERATION = PILOT_GSM8K_GENERATION
    PROMPT_PROBE_SAFE_GENERATION = PILOT_SAFE_GENERATION

    DECODER_COMMONSENSE_GENERATION = LLMGenerationParams(
        max_new_tokens=128,  
        do_sample=True,
        temperature=0.1,
        top_p=0.9,
        repetition_penalty=1.2,
        num_return_sequences=1,
        use_cache=True,
        seed=1234,
        dataset_type="commonsense",
        method="decoder_probe",
        source="tools/commonsense_inference.py line 232-243",
        description="CommonsenseQA常识推理（采样生成，与数据收集完全一致）"
    )

    LCD_GSM8K_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/gsm8k/3_pooled_layer16_pca256_logistic",
        target_layer=16,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="gsm8k",
        model_name="llama3",
        source="lcd_gsm8k_llama3_probe",
        description="GSM8K LLaMA3探针（层16，PCA256）"
    )

    LCD_GSM8K_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/gsm8k/3.1_pooled_layer12_pca256_logistic",
        target_layer=12,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="gsm8k",
        model_name="llama31",
        source="lcd_gsm8k_llama31_probe",
        description="GSM8K LLaMA3.1探针（层12，PCA256）"
    )

    LCD_SAFE_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/safe/3_pooled_layer17_pca256_logistic",
        target_layer=17,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="safe",
        model_name="llama3",
        source="lcd_safe_llama3_probe",
        description="Safe LLaMA3探针（层17，PCA256，pooled）"
    )

    LCD_SAFE_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/safe/3.1_pooled_layer12_pca256_logistic",
        target_layer=12,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="logistic",
        dataset_type="safe",
        model_name="llama31",
        source="lcd_safe_llama31_probe",
        description="Safe LLaMA3.1探针（层12，PCA256，pooled）"
    )

    LCD_COMMONSENSE_LLAMA3 = LCDProbeConfig(
        model_path="lcd_models/commonsense/3_pooled_layer14_pca256_mlp",
        target_layer=14,
        pooling_method="pooled",
        pca_dim=256,
        probe_type="mlp",
        dataset_type="commonsense",
        model_name="llama3",
        source="lcd_commonsense_llama3_probe",
        description="CommonsenseQA LLaMA3探针（层14，PCA256，MLP，pooled）"
    )

    LCD_COMMONSENSE_LLAMA31 = LCDProbeConfig(
        model_path="lcd_models/commonsense/3.1_query_layer14_pca256_mlp",
        target_layer=14,
        pooling_method="query",
        pca_dim=256,
        probe_type="mlp",
        dataset_type="commonsense",
        model_name="llama31",
        source="lcd_commonsense_llama31_probe",
        description="CommonsenseQA LLaMA3.1探针（层14，PCA256，MLP，query）"
    )

    PROMPT_PROBE_CONFIDENCE_PROMPT = (
        "On a scale of 0 to 100, how confident are you in your answer? "
        "Please provide only a number without any explanation."
    )

    GLOBAL_SEED = 1234

    API_BASE_URL = "https://api.openai.com/v1"
    DEFAULT_API_TIMEOUT = 60     
    DEFAULT_API_RETRY = 3

    DEFAULT_CONSENSUS_THRESHOLD = 0.5
    DEFAULT_MAX_ITERATIONS = 3

    CUDA_DETERMINISTIC = True                                      
    CUDA_BENCHMARK = False                                     

    @classmethod
    def get_llm_generation_params(cls, method: str, dataset_type: str) -> LLMGenerationParams:

        method = method.lower().replace('_', '').replace('-', '')
        dataset = dataset_type.lower().strip()

        key = f"{method}_{dataset}_generation"
        key = key.replace('promptprobe', 'prompt_probe').replace('decoderprobe', 'decoder')
        key = key.upper()

        if not hasattr(cls, key):
            raise ValueError(
                f"未找到配置: method={method}, dataset={dataset_type}\n"
                f"尝试的key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def get_lcd_config(cls, dataset_type: str, model_name: str) -> LCDProbeConfig:

        dataset = dataset_type.lower().strip()
        model = model_name.lower().strip()

        key = f"LCD_{dataset.upper()}_{model.upper()}"

        if not hasattr(cls, key):
            raise ValueError(
                f"未找到LCD配置: dataset={dataset_type}, model={model_name}\n"
                f"尝试的key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def get_data_path(cls, method: str, dataset_type: str) -> str:

        method = method.lower().replace('_', '').replace('-', '')
        dataset = dataset_type.lower().strip()

        if method in ['pilot', 'promptprobe']:
            prefix = 'PILOT'
        else:                 
            prefix = 'DECODER'

        key = f"{prefix}_{dataset.upper()}_DATA"

        if not hasattr(cls, key):
            raise ValueError(
                f"未找到数据路径: method={method}, dataset={dataset_type}\n"
                f"尝试的key: {key}"
            )

        return getattr(cls, key)

    @classmethod
    def print_all_configs(cls):

        print("=" * 120)
        print(" 全局参数配置中心 ".center(120, "="))
        print("=" * 120)

        print("\n【模型路径】")
        print(f"  LLAMA3:  {cls.LLAMA3_MODEL_PATH}")
        print(f"  LLAMA31: {cls.LLAMA31_MODEL_PATH}")

        print("\n【数据集路径】")
        print(f"  Decoder GSM8K:        {cls.DECODER_GSM8K_DATA}")
        print(f"  Decoder Safe:         {cls.DECODER_SAFE_DATA}")
        print(f"  Decoder CommonsenseQA: {cls.DECODER_COMMONSENSE_DATA}")
        print(f"  Pilot GSM8K:          {cls.PILOT_GSM8K_DATA}")
        print(f"  Pilot Safe:           {cls.PILOT_SAFE_DATA}")

        print("\n【LLM生成参数】")
        for attr_name in dir(cls):
            if attr_name.endswith('_GENERATION'):
                params = getattr(cls, attr_name)
                if isinstance(params, LLMGenerationParams):
                    mode = "采样" if params.do_sample else "确定性"
                    print(f"\n  {attr_name}:")
                    print(f"    方法: {params.method}, 数据集: {params.dataset_type}")
                    print(f"    模式: {mode}, temp={params.temperature}, max_tokens={params.max_new_tokens}")
                    print(f"    来源: {params.source}")

        print("\n【LCD探针配置】")
        for attr_name in dir(cls):
            if attr_name.startswith('LCD_'):
                config = getattr(cls, attr_name)
                if isinstance(config, LCDProbeConfig):
                    print(f"\n  {attr_name}:")
                    print(f"    数据集: {config.dataset_type}, 模型: {config.model_name}")
                    print(f"    层: {config.target_layer}, 池化: {config.pooling_method}, PCA: {config.pca_dim}")
                    print(f"    路径: {config.model_path}")

        print("\n【其他固定参数】")
        print(f"  全局种子: {cls.GLOBAL_SEED}")
        print(f"  CUDA确定性: {cls.CUDA_DETERMINISTIC}")
        print(f"  CUDA Benchmark: {cls.CUDA_BENCHMARK}")
        print(f"  API Base URL: {cls.API_BASE_URL}")
        print(f"  API超时: {cls.DEFAULT_API_TIMEOUT}s")

        print("\n" + "=" * 120)

    @classmethod
    def validate_all_paths(cls):

        print("\n" + "=" * 120)
        print(" 路径验证 ".center(120, "="))
        print("=" * 120)

        issues = []

        for name in ['LLAMA3_MODEL_PATH', 'LLAMA31_MODEL_PATH']:
            path = Path(getattr(cls, name))
            status = "OK" if path.exists() else "MISSING"
            print(f"[{status}] {name}: {path}")
            if not path.exists():
                issues.append(f"模型路径不存在: {path}")

        for attr_name in dir(cls):
            if attr_name.endswith('_DATA'):
                path = Path(getattr(cls, attr_name))
                status = "OK" if path.exists() else "MISSING"
                print(f"[{status}] {attr_name}: {path}")
                if not path.exists():
                    issues.append(f"数据集路径不存在: {path}")

        for attr_name in dir(cls):
            if attr_name.startswith('LCD_'):
                config = getattr(cls, attr_name)
                if isinstance(config, LCDProbeConfig):
                    path = Path(config.model_path)

                    exists = path.exists() or path.parent.exists()
                    status = "OK" if exists else "MISSING"
                    print(f"[{status}] {attr_name}: {path}")
                    if not exists:
                        issues.append(f"LCD路径不存在: {path}")

        print("\n" + "=" * 120)

        if issues:
            print("发现问题:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("所有路径验证通过!")

        print("=" * 120)

def get_generation_params(dataset_type: str) -> LLMGenerationParams:

    return GlobalParams.get_llm_generation_params('decoder_probe', dataset_type)

if __name__ == "__main__":
    import sys

    GlobalParams.print_all_configs()

    GlobalParams.validate_all_paths()

    print("\n" + "=" * 120)
    print(" 快捷访问测试 ".center(120, "="))
    print("=" * 120)

    try:
        print("\n1. 获取Decoder GSM8K生成参数:")
        params = GlobalParams.get_llm_generation_params('decoder_probe', 'gsm8k')
        print(f"   do_sample={params.do_sample}, temperature={params.temperature}")

        print("\n2. 获取Safe LLaMA3 LCD配置:")
        lcd = GlobalParams.get_lcd_config('safe', 'llama3')
        print(f"   target_layer={lcd.target_layer}, pca_dim={lcd.pca_dim}")

        print("\n3. 获取Pilot GSM8K数据路径:")
        data_path = GlobalParams.get_data_path('pilot', 'gsm8k')
        print(f"   {data_path}")

        print("\n所有测试通过")

    except Exception as e:
        print(f"\n测试失败: {e}")
        sys.exit(1)

    print("=" * 120)

