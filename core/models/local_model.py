
import logging
from typing import Dict, Any, Optional
from .base_model import BaseModel

try:
    from config.unified_config_manager import get_global_config_manager
except ImportError:
    try:
        from config import get_config
    except ImportError:

        def get_config():
            return type('Config', (), {
                'llm_config': {},
                'model_config': {},
                'experiment_config': {}
            })()

logger = logging.getLogger(__name__)

class LocalModel(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.api_base_url = kwargs.get('api_base_url', 'http://localhost:11434')

        self.model_type = kwargs.get('service_type', kwargs.get('model_type', 'ollama'))
        self.stream = kwargs.get('stream', False)

        self._validate_local_config()

        logger.info(f"本地模型初始化完成: {self.model_name}, 服务类型: {self.model_type}, URL: {self.api_base_url}")

    def _validate_local_config(self):

        if not self.api_base_url:
            raise ValueError("本地模型需要提供API基础URL")

        if self.model_type not in ['ollama', 'lm_studio', 'vllm', 'openai_compatible']:
            logger.warning(f"未知的模型类型: {self.model_type}，将使用通用的OpenAI兼容模式")
            self.model_type = 'openai_compatible'

    async def _generate_impl(self, prompt: str, **kwargs) -> str:

        if self.model_type == 'ollama':
            return await self._generate_with_ollama(prompt, **kwargs)
        else:
            return await self._generate_openai_compatible(prompt, **kwargs)

    async def _generate_with_ollama(self, prompt: str, **kwargs) -> str:

        try:

            request_data = {
                "model": self.model_name,
                "prompt": prompt,
                "stream": self.stream,
                "options": {
                    "temperature": kwargs.get('temperature', self.temperature),
                    "num_predict": kwargs.get('max_tokens', self.max_tokens),
                    "top_p": kwargs.get('top_p', 0.9),
                    "top_k": kwargs.get('top_k', 50)
                }
            }

            url = f"{self.api_base_url.rstrip('/')}/api/generate"

            response_data = await self._network_client.post_json(
                url=url,
                data=request_data,
                timeout=kwargs.get('timeout', self.timeout)
            )

            if 'response' in response_data:
                response_text = response_data['response']

                if 'eval_count' in response_data:
                    self._stats['total_tokens'] += response_data.get('eval_count', 0)

                return response_text
            else:
                raise ValueError("Ollama响应格式不正确")

        except Exception as e:
            logger.error(f"Ollama API调用失败: {e}")

            await self._check_ollama_model()
            raise

    async def _generate_openai_compatible(self, prompt: str, **kwargs) -> str:

        try:

            request_data = {
                "model": self.model_name,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "stream": False
            }

            headers = {"Content-Type": "application/json"}

            if self.model_type == 'lm_studio':
                url = f"{self.api_base_url.rstrip('/')}/v1/chat/completions"
            else:
                url = f"{self.api_base_url.rstrip('/')}/chat/completions"

            response_data = await self._network_client.post_json(
                url=url,
                data=request_data,
                headers=headers,
                timeout=kwargs.get('timeout', self.timeout)
            )

            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']

                if 'usage' in response_data:
                    self._stats['total_tokens'] += response_data['usage'].get('total_tokens', 0)

                return content
            else:
                raise ValueError("API响应格式不正确")

        except Exception as e:
            logger.error(f"本地API调用失败: {e}")
            raise

    async def _check_ollama_model(self):

        try:

            url = f"{self.api_base_url.rstrip('/')}/api/tags"
            response_data = await self._network_client.post_json(url, {})

            if 'models' in response_data:
                available_models = [model['name'] for model in response_data['models']]
                if self.model_name not in available_models:
                    logger.warning(f"模型 {self.model_name} 不在可用列表中: {available_models}")
                    logger.info(f"请运行: ollama pull {self.model_name}")
                else:
                    logger.info(f"模型 {self.model_name} 可用")

        except Exception as e:
            logger.error(f"检查Ollama模型失败: {e}")

    async def test_connection(self) -> bool:

        try:
            if self.model_type == 'ollama':

                url = f"{self.api_base_url.rstrip('/')}/api/version"
                await self._network_client.post_json(url, {})
                logger.info("Ollama服务连接正常")

                await self._check_ollama_model()

            else:

                test_response = await self.generate("测试", max_tokens=10)
                if not test_response:
                    return False
                logger.info("本地模型服务连接正常")

            return True

        except Exception as e:
            logger.error(f"本地模型连接测试失败: {e}")
            return False

    async def list_available_models(self) -> list:

        try:
            if self.model_type == 'ollama':
                url = f"{self.api_base_url.rstrip('/')}/api/tags"
                response_data = await self._network_client.post_json(url, {})

                if 'models' in response_data:
                    return [
                        {
                            'name': model['name'],
                            'size': model.get('size', 0),
                            'modified_at': model.get('modified_at', ''),
                            'digest': model.get('digest', '')
                        }
                        for model in response_data['models']
                    ]

            else:

                logger.warning(f"模型类型 {self.model_type} 不支持列表功能")
                return []

        except Exception as e:
            logger.error(f"获取模型列表失败: {e}")
            return []

    async def pull_model(self, model_name: Optional[str] = None) -> bool:

        if self.model_type != 'ollama':
            logger.warning("只有Ollama支持模型拉取功能")
            return False

        target_model = model_name or self.model_name

        try:
            url = f"{self.api_base_url.rstrip('/')}/api/pull"
            request_data = {"name": target_model}

            logger.info(f"开始拉取模型: {target_model}")
            await self._network_client.post_json(url, request_data, timeout=300)          

            logger.info(f"模型 {target_model} 拉取成功")
            return True

        except Exception as e:
            logger.error(f"拉取模型失败: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:

        base_info = super().get_model_info()

        local_info = {
            'model_type': 'Local',
            'service_type': self.model_type,
            'api_base_url': self.api_base_url,
            'stream_enabled': self.stream,
            'cost_estimate': {
                'total_tokens': self._stats['total_tokens'],
                'estimated_cost': 0.0,          
                'currency': 'Free'
            }
        }

        return {**base_info, **local_info}

    def get_usage_instructions(self) -> Dict[str, str]:

        instructions = {
            'ollama': '''
Ollama使用说明:
1. 安装Ollama: curl -fsSL https://ollama.ai/install.sh | sh
2. 启动服务: ollama serve
3. 拉取模型: ollama pull {model_name}
4. 验证模型: ollama list
            ''',
            'lm_studio': '''
LM Studio使用说明:
1. 下载并安装LM Studio
2. 在LM Studio中加载模型
3. 启动本地服务器 (通常在端口1234)
4. 确保API兼容模式已开启
            ''',
            'vllm': '''
vLLM使用说明:
1. 安装vLLM: pip install vllm
2. 启动API服务器: python -m vllm.entrypoints.openai.api_server --model {model_name}
3. 服务通常运行在端口8000
            '''
        }

        return {
            'service_type': self.model_type,
            'instructions': instructions.get(self.model_type, '请参考相应服务的文档'),
            'api_url': self.api_base_url,
            'current_model': self.model_name
        } 