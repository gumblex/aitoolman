import abc
import logging
from typing import Optional, List, Dict, Any, Union

from . import util
from .channel import ChannelWriter
from .model import LLMProviderRequest, Message, ModelInfo, LLMNoAvailableModelError
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

    def make_request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Dict[str, Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            output_channel: Optional[ChannelWriter] = None
    ) -> LLMProviderRequest:
        """新建 LLMProviderRequest 对象"""
        return LLMProviderRequest(
            self.client_id, context_id,
            util.get_id(),
            model_name, messages, tools or [], options or {},
            stream, output_channel
        )

    @abc.abstractmethod
    async def request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Optional[Dict[str, Dict[str, Any]]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            output_channel: Optional[ChannelWriter] = None
    ) -> LLMProviderRequest:
        """实际调用 LLM"""
        pass

    @abc.abstractmethod
    async def cancel(self, request_id: str):
        pass

    @abc.abstractmethod
    async def audit_event(self, context_id: str, event_type: str, **kwargs):
        """记录用户自定义审计事件，kwargs 应能序列化为 JSON"""
        pass

    @abc.abstractmethod
    async def list_models(self, tag: Optional[str] = None) -> List[ModelInfo]:
        """列出匹配的可用模型

        Args:
            tag: 可选标签/模型名/别名，无tag返回所有可用模型

        Returns:
            ModelInfo列表
        """
        pass

    @abc.abstractmethod
    async def resolve_model(self, tags: Union[str, List[str]], messages: Optional[List[Message]] = None) -> List[str]:
        """解析出最终使用的真实模型名

        Args:
            tags: 标签/模型名/别名，支持单个字符串或列表
            messages: 可选消息列表，用于Token数估算过滤

        Returns:
            真实模型名

        Raises:
            LLMNoAvailableModelError: 无可用模型
        """
        pass

    async def update_config(self, new_config: Dict[str, Any]):
        """热更新全量配置"""
        raise NotImplementedError

    async def update_api_config(self, model_name: str, model_config: Dict[str, Any]):
        """更新单个模型配置"""
        raise NotImplementedError

    async def change_api_status(self, model_name: str, enabled: bool):
        """启用/禁用模型"""
        raise NotImplementedError

    async def update_model_tag(self, tag: str, models: List[str]):
        """更新 model_tag"""
        raise NotImplementedError


class LLMLocalClient(LLMClient):
    """本地客户端，直接调用LLMProviderManager"""

    def __init__(self, config: Dict[str, Any]):
        super().__init__()
        self.provider_manager = LLMProviderManager(config)

    async def request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Optional[Dict[str, Dict[str, Any]]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            output_channel: Optional[ChannelWriter] = None
    ) -> LLMProviderRequest:
        request = self.make_request(
            model_name, messages, tools, options, stream,
            context_id, output_channel
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

    async def list_models(self, tag: Optional[str] = None) -> List[ModelInfo]:
        return self.provider_manager.list_models(tag)

    async def resolve_model(self, tags: Union[str, List[str]], messages: Optional[List[Message]] = None) -> List[str]:
        if not tags:
            raise LLMNoAvailableModelError("No tags provided.")
        if isinstance(tags, str):
            tags = [tags]
        return self.provider_manager.resolve_model(tags, messages)

    async def update_config(self, new_config: Dict[str, Any]):
        self.provider_manager.update_config(new_config)

    async def update_api_config(self, model_name: str, model_config: Dict[str, Any]):
        self.provider_manager.update_api_config(model_name, model_config)

    async def change_api_status(self, model_name: str, enabled: bool):
        self.provider_manager.change_api_status(model_name, enabled)

    async def update_model_tag(self, tag: str, models: List[str]):
        self.provider_manager.update_model_tag(tag, models)
