
import asyncio
import logging
import time
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
import aiohttp
import json

try:
    from config.unified_config_manager import get_global_config_manager
except ImportError:
    get_global_config_manager = None

logger = logging.getLogger(__name__)

class NetworkClient:

    def __init__(self, timeout: int = 30):

        self.timeout = timeout
        self._session: Optional[aiohttp.ClientSession] = None

        self._connector = aiohttp.TCPConnector(
            limit=100,         
            ttl_dns_cache=300,           
            use_dns_cache=True,
        )

    async def get_session(self) -> aiohttp.ClientSession:

        if self._session is None or self._session.closed:
            timeout = aiohttp.ClientTimeout(total=self.timeout)

            import os
            use_proxy = bool(
                os.environ.get('HTTP_PROXY') or os.environ.get('http_proxy') or
                os.environ.get('HTTPS_PROXY') or os.environ.get('https_proxy')
            )
            if use_proxy:
                logger.info("检测到代理环境变量，启用代理转发 (trust_env=True)")
            self._session = aiohttp.ClientSession(
                connector=self._connector,
                timeout=timeout,
                trust_env=use_proxy,                       
                headers={
                    'User-Agent': 'Byzantine-V2-Research/1.0',
                    'Content-Type': 'application/json'
                }
            )
        return self._session

    async def post_json(self, url: str, data: Dict[str, Any], 
                       headers: Optional[Dict[str, str]] = None,
                       timeout: Optional[int] = None) -> Dict[str, Any]:

        session = await self.get_session()
        request_headers = headers or {}
        request_timeout = timeout or self.timeout

        try:
            async with session.post(
                url, 
                json=data, 
                headers=request_headers,
                timeout=aiohttp.ClientTimeout(total=request_timeout)
            ) as response:
                if response.status >= 400:

                    try:
                        error_text = await response.text()
                    except Exception:
                        error_text = ''
                    logger.error(f"HTTP {response.status} 调用失败: {url}\n请求体: {data}\n响应体: {error_text}")
                    response.raise_for_status()
                return await response.json()
        except aiohttp.ClientError as e:
            logger.error(f"网络请求失败: {url}, 错误: {e}")
            raise
        except json.JSONDecodeError as e:
            logger.error(f"JSON解析失败: {e}")
            raise

    async def close(self):

        if self._session and not self._session.closed:
            await self._session.close()

class BaseModel(ABC):

    def __init__(self, **kwargs):

        self.config = {}
        try:
            if get_global_config_manager is not None:
                cfg_mgr = get_global_config_manager()

                self.config = {
                    'llm_config': {},
                    'model_config': {},
                    'experiment_config': {}
                }
            else:
                self.config = {
                    'llm_config': {},
                    'model_config': {},
                    'experiment_config': {}
                }
        except Exception:
            self.config = {
                'llm_config': {},
                'model_config': {},
                'experiment_config': {}
            }

        self.model_name = kwargs.get('model_name', 'unknown')
        self.temperature = kwargs.get('temperature', 0.1)
        self.max_tokens = kwargs.get('max_tokens', 500)

        self.top_p = kwargs.get('top_p', 0.9)
        self.frequency_penalty = kwargs.get('frequency_penalty', 0.0)
        self.presence_penalty = kwargs.get('presence_penalty', 0.0)
        self.timeout = kwargs.get('timeout', 30)
        self.max_retries = kwargs.get('max_retries', 3)
        self.retry_delay = kwargs.get('retry_delay', 1.0)
        self.seed = kwargs.get('seed', 1234)              

        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_response_time': 0.0,
            'last_call_time': None
        }

        self._network_client = NetworkClient(timeout=self.timeout)

        self._initialized = True
        logger.info(f"模型 {self.model_name} 初始化完成，准备调用")

    @abstractmethod
    async def _generate_impl(self, prompt: str, **kwargs) -> str:

        pass

    async def generate(self, prompt: str, **kwargs) -> str:

        if not self._initialized:
            raise RuntimeError("模型未正确初始化")

        start_time = time.time()
        self._stats['total_calls'] += 1
        self._stats['last_call_time'] = start_time

        try:

            processed_prompt = self._preprocess_prompt(prompt)
            processed_kwargs = self._preprocess_kwargs(kwargs)

            response = await self._generate_impl(processed_prompt, **processed_kwargs)

            final_response = self._postprocess_response(response)

            response_time = time.time() - start_time
            self._stats['successful_calls'] += 1
            self._stats['total_response_time'] += response_time

            logger.debug(f"模型调用成功: {self.model_name}, 用时: {response_time:.2f}s")
            return final_response

        except Exception as e:
            self._stats['failed_calls'] += 1
            logger.error(f"模型调用失败: {self.model_name}, 错误: {e}")
            raise

    async def generate_with_messages(self, messages: List[Dict[str, str]], **kwargs) -> str:

        prompt_parts = []

        for message in messages:
            role = message.get("role", "user")
            content = message.get("content", "")

            if role == "system":
                prompt_parts.append(f"System: {content}")
            elif role == "user":
                prompt_parts.append(f"User: {content}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {content}")

        combined_prompt = "\n\n".join(prompt_parts)
        return await self.generate(combined_prompt, **kwargs)

    async def generate_with_retry(self, prompt: str, **kwargs) -> str:

        last_exception = None

        for attempt in range(self.max_retries):
            try:
                return await self.generate(prompt, **kwargs)
            except Exception as e:
                last_exception = e
                if attempt < self.max_retries - 1:
                    delay = self.retry_delay * (2 ** attempt)        
                    logger.warning(f"第 {attempt + 1} 次尝试失败，{delay}s 后重试: {e}")
                    await asyncio.sleep(delay)
                else:
                    logger.error(f"重试 {self.max_retries} 次后仍然失败")

        raise last_exception

    def _preprocess_prompt(self, prompt: str) -> str:

        if not prompt or not prompt.strip():
            raise ValueError("提示文本不能为空")
        return prompt.strip()

    def _preprocess_kwargs(self, kwargs: Dict[str, Any]) -> Dict[str, Any]:

        processed = kwargs.copy()

        processed.setdefault('temperature', self.temperature)
        processed.setdefault('max_tokens', self.max_tokens)
        processed.setdefault('seed', self.seed)
        processed.setdefault('top_p', self.top_p)
        processed.setdefault('frequency_penalty', self.frequency_penalty)
        processed.setdefault('presence_penalty', self.presence_penalty)

        return processed

    def _postprocess_response(self, response: str) -> str:

        if not response:
            return ""
        return response.strip()

    async def test_connection(self) -> bool:

        try:
            test_response = await self.generate("测试", max_tokens=10)
            return bool(test_response)
        except Exception as e:
            logger.error(f"连接测试失败: {e}")
            return False

    def get_model_info(self) -> Dict[str, Any]:

        success_rate = 0.0
        avg_response_time = 0.0

        if self._stats['total_calls'] > 0:
            success_rate = self._stats['successful_calls'] / self._stats['total_calls']

        if self._stats['successful_calls'] > 0:
            avg_response_time = self._stats['total_response_time'] / self._stats['successful_calls']

        return {
            'model_name': self.model_name,
            'total_calls': self._stats['total_calls'],
            'successful_calls': self._stats['successful_calls'],
            'failed_calls': self._stats['failed_calls'],
            'success_rate': success_rate,
            'avg_response_time': avg_response_time,
            'total_tokens': self._stats['total_tokens'],
            'last_call_time': self._stats['last_call_time'],
            'config': {
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                    'timeout': self.timeout,
                    'max_retries': self.max_retries,
                    'top_p': self.top_p,
                    'frequency_penalty': self.frequency_penalty,
                    'presence_penalty': self.presence_penalty
            }
        }

    def reset_stats(self):

        self._stats = {
            'total_calls': 0,
            'successful_calls': 0,
            'failed_calls': 0,
            'total_tokens': 0,
            'total_response_time': 0.0,
            'last_call_time': None
        }
        logger.info(f"模型 {self.model_name} 统计数据已重置")

    async def close(self):

        await self._network_client.close()
        logger.info(f"模型 {self.model_name} 已关闭")

    def __del__(self):

        if hasattr(self, '_network_client'):
            try:

                loop = asyncio.get_event_loop()
                if loop.is_running():
                    loop.create_task(self._network_client.close())
                else:
                    loop.run_until_complete(self._network_client.close())
            except:
                pass 