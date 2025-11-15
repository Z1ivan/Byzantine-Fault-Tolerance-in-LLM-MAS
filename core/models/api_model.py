
import os
import logging
from typing import Dict, Any, Optional, List
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

class APIModel(BaseModel):

    def __init__(self, **kwargs):

        super().__init__(**kwargs)

        self.api_key = kwargs.get('api_key') or self._get_api_key()

        self.api_base_url = kwargs.get('api_base_url', 'https://api.openai.com/v1')
        self.use_camel = kwargs.get('use_camel', False)                            

        if not self.api_key:
            raise ValueError("API密钥未提供，请设置环境变量或传入api_key参数")

        if self.use_camel:
            self._prepare_camel_config()

        logger.info(f"API模型初始化完成: {self.model_name}, 基础URL: {self.api_base_url}")

    def _get_api_key(self) -> Optional[str]:

        try:
            if hasattr(self.config, 'llm_config') and self.config.llm_config:
                return self.config.llm_config.get('api_key')
            elif isinstance(self.config, dict) and 'api_key' in self.config:
                return self.config['api_key']
        except:
            pass

        api_keys = [
            'OPENAI_API_KEY',
            'OPENAI_COMPATIBILITY_API_KEY', 
            'API_KEY'
        ]

        for key_name in api_keys:
            api_key = os.getenv(key_name)
            if api_key:
                logger.debug(f"从环境变量 {key_name} 获取API密钥")
                return api_key

        return None

    def _prepare_camel_config(self):

        self._camel_config = {
            'model_name': self.model_name,
            'api_key': self.api_key,
            'api_base_url': self.api_base_url,
            'temperature': self.temperature,
            'max_tokens': self.max_tokens,
            'timeout': self.timeout
        }
        logger.debug("Camel框架配置准备完成")

    async def _generate_impl(self, prompt: str, **kwargs) -> str:

        if self.use_camel:
            return await self._generate_with_camel(prompt, **kwargs)
        else:
            return await self._generate_direct_api(prompt, **kwargs)

    def _get_effective_model_name(self, raw_model_name: Optional[str] = None) -> str:

        model_name = (raw_model_name or self.model_name or "").strip()
        return model_name

    async def _generate_with_camel(self, prompt: str, **kwargs) -> str:

        try:

            from camel.models import ModelFactory
            from camel.messages import BaseMessage
            from camel.types import ModelPlatformType

            if not hasattr(self, '_camel_config'):
                self._prepare_camel_config()

            model_name_lower = self.model_name.lower()

            if 'gpt-4' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                if 'mini' in model_name_lower:
                    model_type = "gpt-4o-mini"
                else:
                    model_type = "gpt-4o"
            elif 'gpt-3.5' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = "gpt-3.5-turbo"
            elif 'deepseek' in model_name_lower:
                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = self.model_name
            else:

                model_platform = ModelPlatformType.OPENAI_COMPATIBLE_MODEL
                model_type = self.model_name

            model_config = {
                'temperature': kwargs.get('temperature', self.temperature),
                'max_tokens': kwargs.get('max_tokens', self.max_tokens),
                'seed': kwargs.get('seed', self.seed),
            }

            model = ModelFactory.create(
                model_platform=model_platform,
                model_type=model_type,
                api_key=self.api_key,
                url=self.api_base_url,
                model_config_dict=model_config
            )

            message = BaseMessage.make_user_message(
                role_name="user",
                content=prompt
            )

            response = model.run([message])

            if hasattr(response, 'content'):
                return response.content
            elif hasattr(response, 'msg'):
                return response.msg.content
            else:
                return str(response)

        except ImportError:
            logger.warning("Camel框架未安装，回退到直接API调用")
            return await self._generate_direct_api(prompt, **kwargs)
        except Exception as e:
            logger.error(f"Camel框架调用失败: {e}")

            return await self._generate_direct_api(prompt, **kwargs)

    async def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:

        if self.use_camel:

            return await super().generate_with_messages(messages, **kwargs)
        else:

            return await self._generate_direct_api_with_messages(messages, **kwargs)

    async def _generate_direct_api(self, prompt: str, **kwargs) -> str:

        try:

            effective_model = self._get_effective_model_name()
            request_data = {
                "model": effective_model,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "top_p": kwargs.get('top_p', getattr(self, 'top_p', 0.9)),
                "frequency_penalty": kwargs.get('frequency_penalty', getattr(self, 'frequency_penalty', 0.0)),
                "presence_penalty": kwargs.get('presence_penalty', getattr(self, 'presence_penalty', 0.0)),
                "seed": kwargs.get('seed', self.seed),
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

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
            logger.error(f"直接API调用失败: {e}")
            raise

    async def _generate_direct_api_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:

        try:

            effective_model = self._get_effective_model_name()
            request_data = {
                "model": effective_model,
                "messages": messages,
                "temperature": kwargs.get('temperature', self.temperature),
                "max_tokens": kwargs.get('max_tokens', self.max_tokens),
                "top_p": kwargs.get('top_p', getattr(self, 'top_p', 0.9)),
                "frequency_penalty": kwargs.get('frequency_penalty', getattr(self, 'frequency_penalty', 0.0)),
                "presence_penalty": kwargs.get('presence_penalty', getattr(self, 'presence_penalty', 0.0)),
                "seed": kwargs.get('seed', self.seed),
                "stream": False
            }

            headers = {
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json"
            }

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
            logger.error(f"使用消息列表的API调用失败: {e}")
            raise

    def get_cost_estimate(self) -> Dict[str, Any]:

        cost_per_1k_tokens = {
            'gpt-4o-mini': 0.0015,
            'gpt-3.5-turbo': 0.002,
            'deepseek-chat': 0.0014,
        }

        base_cost = cost_per_1k_tokens.get(self.model_name, 0.002)
        total_cost = (self._stats['total_tokens'] / 1000) * base_cost

        return {
            'total_tokens': self._stats['total_tokens'],
            'estimated_cost': total_cost,
            'cost_per_1k_tokens': base_cost,
            'currency': 'USD'
        }

    def get_model_info(self) -> Dict[str, Any]:

        base_info = super().get_model_info()

        api_info = {
            'model_type': 'API',
            'api_base_url': self.api_base_url,
            'use_camel': self.use_camel,
            'cost_estimate': self.get_cost_estimate()
        }

        return {**base_info, **api_info} 