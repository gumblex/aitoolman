import abc
import logging
from typing import Optional, List, Dict, Any, Callable

from . import util
from .channel import TextChannel
from .model import LLMRequest, Message
from .provider import LLMProviderManager

logger = logging.getLogger(__name__)


class LLMClient(abc.ABC):
    """LLM客户端抽象基类"""

    def __init__(self):
        self.client_id = util.get_host_id()

    async def __aenter__(self):
        await self.initialize()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

    async def initialize(self):
        pass

    async def close(self):
        pass

    def _make_request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Dict[str, Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            response_channel: Optional[TextChannel] = None,
            reasoning_channel: Optional[TextChannel] = None
    ) -> LLMRequest:
        """新建 LLMRequest 对象"""
        return LLMRequest(
            self.client_id, context_id,
            util.get_id(),
            model_name, messages, tools or [], options or {},
            stream, response_channel, reasoning_channel
        )

    @abc.abstractmethod
    async def request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Dict[str, Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            response_channel: Optional[TextChannel] = None,
            reasoning_channel: Optional[TextChannel] = None
    ) -> LLMRequest:
        """实际调用 LLM"""
        pass

    @abc.abstractmethod
    async def cancel(self, request_id: str):
        pass

    @abc.abstractmethod
    async def audit_event(self, context_id: str, event_type: str, **kwargs):
        """记录用户自定义审计事件"""
        pass


class LLMLocalClient(LLMClient):
    """本地客户端，直接调用LLMProviderManager"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.provider_manager = LLMProviderManager(config)

    async def request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Dict[str, Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            response_channel: Optional[TextChannel] = None,
            reasoning_channel: Optional[TextChannel] = None
    ) -> LLMRequest:
        request = self._make_request(
            model_name, messages, tools, options, stream,
            context_id, response_channel, reasoning_channel
        )
        self.provider_manager.process_request(request)
        return request

    async def cancel(self, request_id: str):
        await self.provider_manager.cancel_request(request_id)

    async def initialize(self):
        await self.provider_manager.initialize()

    async def close(self):
        await self.provider_manager.cleanup()

    async def audit_event(self, context_id: str, event_type: str, **kwargs):
        logger.info(
            "[AUDIT: context %s] event: %s, %s",
            context_id, event_type, kwargs
        )
