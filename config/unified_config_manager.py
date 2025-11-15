
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union, Type
from pathlib import Path
from dataclasses import dataclass, asdict
from enum import Enum

from .base_config import BaseConfig, BaseConfigManager
from .prompt_probe_config import PromptProbeConfig, PromptProbeConfigManager
from .decoder_probe_config import DecoderProbeConfig

logger = logging.getLogger(__name__)

class MethodType(Enum):

    PROMPT_PROBE = "prompt_probe"
    PILOT = "pilot"
    BASE = "base"
    DECODER = "decoder_probe"

_REPO_ROOT = Path(__file__).resolve().parents[1]

@dataclass
class ProjectConfig:

    project_name: str = "Byzantine Fault Tolerance System"
    project_version: str = "2.0.0"
    project_root: str = str(_REPO_ROOT)

    config_dir: str = str(_REPO_ROOT / "config")
    core_dir: str = str(_REPO_ROOT / "core")
    methods_dir: str = str(_REPO_ROOT / "methods")
    dataset_dir: str = str(_REPO_ROOT / "data" / "byzantine")

    models_base_dir: str = str(_REPO_ROOT / "models")
    llama3_model_path: str = str(_REPO_ROOT / "models" / "LLama-3-8B-Instruct")
    llama31_model_path: str = str(_REPO_ROOT / "models" / "LLama-3.1-8B-Instruct")

    default_api_base_url: str = "https://api.openai.com/v1"
    default_temperature: float = 0.1
    default_max_tokens: int = 1000
    default_seed: int = 1234

    results_base_dir: str = str(_REPO_ROOT / "results")
    logs_dir: str = str(_REPO_ROOT / "logs")
    cache_dir: str = str(_REPO_ROOT / "cache")

    environment: str = "development"                                    
    debug_mode: bool = True
    log_level: str = "INFO"

    max_workers: int = 4
    timeout_seconds: int = 300
    retry_attempts: int = 3

class UnifiedConfigManager:

    CONFIG_CLASSES = {
        MethodType.BASE: BaseConfig,
        MethodType.PROMPT_PROBE: PromptProbeConfig,
        MethodType.PILOT: BaseConfig,                           
        MethodType.DECODER: DecoderProbeConfig,
    }

    CONFIG_MANAGERS = {
        MethodType.BASE: BaseConfigManager,
        MethodType.PROMPT_PROBE: PromptProbeConfigManager,
        MethodType.PILOT: BaseConfigManager,                            
        MethodType.DECODER: BaseConfigManager,                       
    }

    def __init__(self, project_config: Optional[ProjectConfig] = None):

        self.project_config = project_config or ProjectConfig()
        self._config_cache: Dict[MethodType, Any] = {}
        self._manager_cache: Dict[MethodType, Any] = {}

        self._setup_logging()

        self._validate_project_config()

        logger.info("统一配置管理器初始化完成")

    def _setup_logging(self):

        log_level = getattr(logging, self.project_config.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=log_level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )

    def _validate_project_config(self):

        errors = []

        paths_to_check = [
            (self.project_config.project_root, "项目根目录"),
            (self.project_config.config_dir, "配置目录"),
            (self.project_config.core_dir, "核心代码目录"),
            (self.project_config.methods_dir, "方法目录"),
        ]

        for path, name in paths_to_check:
            if not Path(path).exists():
                errors.append(f"{name}不存在: {path}")

        model_paths = [
            (self.project_config.llama3_model_path, "LLaMA3模型路径"),
            (self.project_config.llama31_model_path, "LLaMA3.1模型路径"),
        ]

        for path, name in model_paths:
            if not Path(path).exists():
                logger.warning(f"[WARN] {name}不存在: {path}")

        if errors:
            raise ValueError(f"项目配置验证失败: {'; '.join(errors)}")

        logger.info("项目配置验证通过")

    def get_config_manager(self, method_type: MethodType) -> BaseConfigManager:

        if method_type not in self._manager_cache:
            manager_class = self.CONFIG_MANAGERS.get(method_type)
            if not manager_class:
                raise ValueError(f"不支持的方法类型: {method_type}")

            self._manager_cache[method_type] = manager_class()
            logger.debug(f"创建配置管理器: {method_type.value}")

        return self._manager_cache[method_type]

    def load_config(self, method_type: MethodType, 
                   config_file: Optional[str] = None,
                   use_defaults: bool = False,
                   **kwargs) -> Any:

        logger.info(f"加载配置: {method_type.value}")

        cache_key = f"{method_type.value}_{use_defaults}_{bool(kwargs)}"
        if cache_key in self._config_cache:
            logger.debug(f"从缓存获取配置: {method_type.value}")
            return self._config_cache[cache_key]

        manager = self.get_config_manager(method_type)

        if config_file:
            config = self._load_config_from_file(method_type, config_file)
        elif use_defaults:

            config_class = self.CONFIG_CLASSES.get(method_type)
            if not config_class:
                raise ValueError(f"不支持的方法类型: {method_type}")
            config = config_class()
        else:

            parser = manager.create_parser()
            args = parser.parse_args()
            config = manager.parse_args_to_config(args)

        if kwargs:
            config = self._apply_config_overrides(config, kwargs)

        config = self._apply_project_config(config)

        errors = manager.validate_config(config)
        if errors:
            raise ValueError(f"{method_type.value}配置验证失败: {'; '.join(errors)}")

        self._config_cache[cache_key] = config

        logger.info(f"配置加载完成: {method_type.value}")
        return config

    def _load_config_from_file(self, method_type: MethodType, config_file: str) -> Any:

        config_path = Path(config_file)
        if not config_path.exists():
            raise FileNotFoundError(f"配置文件不存在: {config_file}")

        logger.info(f"从文件加载配置: {config_file}")

        if config_path.suffix == '.json':
            with open(config_path, 'r', encoding='utf-8') as f:
                config_data = json.load(f)
        else:
            raise ValueError(f"不支持的配置文件格式: {config_path.suffix}")

        config_class = self.CONFIG_CLASSES.get(method_type)
        if not config_class:
            raise ValueError(f"不支持的方法类型: {method_type}")

        return config_class(**config_data)

    def _apply_config_overrides(self, config: Any, overrides: Dict[str, Any]) -> Any:

        logger.debug(f"应用配置覆盖: {len(overrides)} 个参数")

        config_dict = asdict(config)
        config_dict.update(overrides)

        return type(config)(**config_dict)

    def _apply_project_config(self, config: Any) -> Any:

        logger.debug("应用项目级配置")

        if hasattr(config, 'llama3_model_path'):
            config.llama3_model_path = self.project_config.llama3_model_path
        if hasattr(config, 'llama31_model_path'):
            config.llama31_model_path = self.project_config.llama31_model_path

        if hasattr(config, 'api_base_url') and not config.api_base_url:
            config.api_base_url = self.project_config.default_api_base_url

        if hasattr(config, 'temperature') and config.temperature is None:
            config.temperature = self.project_config.default_temperature
        if hasattr(config, 'max_tokens') and config.max_tokens == 0:
            config.max_tokens = self.project_config.default_max_tokens
        if hasattr(config, 'seed') and config.seed == 0:
            config.seed = self.project_config.default_seed

        return config

    def save_config(self, method_type: MethodType, config: Any, 
                   output_file: Optional[str] = None) -> str:

        if not output_file:
            config_dir = Path(self.project_config.config_dir)
            output_file = config_dir / f"{method_type.value}_config.json"

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        config_dict = asdict(config)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(config_dict, f, indent=2, ensure_ascii=False, default=str)

        logger.info(f"配置已保存: {output_path}")
        return str(output_path)

    def get_all_configs(self) -> Dict[MethodType, Any]:

        configs = {}
        for method_type in MethodType:
            try:
                configs[method_type] = self.load_config(method_type)
            except Exception as e:
                logger.warning(f"加载{method_type.value}配置失败: {e}")

        return configs

    def validate_all_configs(self) -> Dict[MethodType, List[str]]:

        validation_results = {}

        for method_type in MethodType:
            try:

                config = self.load_config(method_type, use_defaults=True)
                manager = self.get_config_manager(method_type)
                errors = manager.validate_config(config)
                validation_results[method_type] = errors
            except Exception as e:
                validation_results[method_type] = [str(e)]

        return validation_results

    def print_project_summary(self):

        print("=" * 60)
        print(f"{self.project_config.project_name} v{self.project_config.project_version}")
        print("=" * 60)
        print(f"项目根目录: {self.project_config.project_root}")
        print(f"配置目录: {self.project_config.config_dir}")
        print(f"环境: {self.project_config.environment}")
        print(f"调试模式: {'开启' if self.project_config.debug_mode else '关闭'}")
        print(f"日志级别: {self.project_config.log_level}")
        print()

        print("支持的方法:")
        for method_type in MethodType:
            print(f"  - {method_type.value}")
        print()

        print("全局模型路径:")
        print(f"  - LLaMA3: {self.project_config.llama3_model_path}")
        print(f"  - LLaMA3.1: {self.project_config.llama31_model_path}")
        print("=" * 60)

_global_config_manager: Optional[UnifiedConfigManager] = None

def get_global_config_manager() -> UnifiedConfigManager:

    global _global_config_manager
    if _global_config_manager is None:
        _global_config_manager = UnifiedConfigManager()
    return _global_config_manager

def load_method_config(method_type: Union[str, MethodType], **kwargs) -> Any:

    if isinstance(method_type, str):
        method_type = MethodType(method_type)

    manager = get_global_config_manager()
    return manager.load_config(method_type, **kwargs)

def validate_project_configs() -> bool:

    manager = get_global_config_manager()
    validation_results = manager.validate_all_configs()

    all_valid = True
    for method_type, errors in validation_results.items():
        if errors:
            print(f"{method_type.value}配置验证失败:")
            for error in errors:
                print(f"   - {error}")
            all_valid = False
        else:
            print(f"{method_type.value}配置验证通过")

    return all_valid

__all__ = [
    'MethodType',
    'ProjectConfig', 
    'UnifiedConfigManager',
    'get_global_config_manager',
    'load_method_config',
    'validate_project_configs'
]

if __name__ == "__main__":

    print("测试统一配置管理器")

    manager = UnifiedConfigManager()
    manager.print_project_summary()

    print("\n验证所有配置:")
    if validate_project_configs():
        print("\n所有配置验证通过！")
    else:
        print("\n部分配置存在问题，请检查。")
