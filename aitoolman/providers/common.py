"""LLM提供商格式策略公共模块：抽象基类、公共辅助函数与统一映射表"""
import abc
import typing
from typing import Optional, Dict, List, Any

import msgspec

from .. import sse
from ..model import LLMProviderRequest, LLMProviderResponse, Message, ToolCall


class ProviderStreamEvent(typing.NamedTuple):
    is_end: bool
    content: str
    reasoning: str
    tool_calls: Any
    response_message: Optional[List[Dict[str, Any]]]

    @classmethod
    def empty(cls, is_end=False):
        return cls(is_end, '', '', [], None)


# 提供商原生 finish/stop reason → 统一 FinishReason 值映射
FINISH_REASON_MAP: Dict[str, str] = {
    # OpenAI finish_reason
    'stop': 'stop',
    'length': 'length',
    'tool_calls': 'tool_calls',
    'function_call': 'tool_calls',
    'content_filter': 'content_filter',
    # Anthropic stop_reason
    'end_turn': 'stop',
    'max_tokens': 'length',
    'stop_sequence': 'stop',
    'tool_use': 'tool_calls',
    'refusal': 'content_filter',
    'pause_turn': 'stop',
    'model_context_window_exceeded': 'length',
}


def map_finish_reason(reason: Optional[str]) -> Optional[str]:
    """将提供商原生 finish/stop reason 映射为统一 FinishReason 值，未知原因返回 None"""
    if reason is None:
        return None
    return FINISH_REASON_MAP.get(reason)


def serialize_tool_parameters(param_configs: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """将配置格式的工具参数解析为 JSON Schema parameters 结构"""
    parameters: Dict[str, Any] = {
        "type": "object",  # 参数结构固定为object类型（符合JSON Schema要求）
        "properties": {},
        "required": []
    }
    for param_name, param_info in param_configs.items():
        param = {"type": param_info.get("type", "string")}
        desc = param_info.get("description", "")
        if desc:
            param["description"] = desc
        parameters["properties"][param_name] = param
        if param_info.get("required"):
            parameters["required"].append(param_name)
    return parameters


def struct_meta_merge(source, field_names, buffer: Dict[str, Any]) -> None:
    """按字段顺序将 Struct 中非 None 的字段合并入缓冲（Struct 值转为纯 dict，剔除值为 None 的键）"""
    for name in field_names:
        value = getattr(source, name, None)
        if value is None:
            continue
        if isinstance(value, msgspec.Struct):
            value = {k: v for k, v in msgspec.structs.asdict(value).items() if v is not None}
        buffer[name] = value


class LLMFormatStrategy(abc.ABC):
    """LLM请求/响应格式策略抽象类"""
    def __init__(self, model_config: Dict[str, Any]):
        self.model_config = model_config  # 模型配置（包含api_type、url、headers等）
        # 流式解析公共缓冲
        self.stream_content_buffer = ''
        self.stream_reasoning_buffer = ''
        self.stream_tool_buffers: List[Dict[str, Any]] = []

    def clear_stream_buffers(self) -> None:
        """清空流式解析缓冲"""
        self.stream_content_buffer = ''
        self.stream_reasoning_buffer = ''
        self.stream_tool_buffers = []

    def build_stream_response_message(self) -> Dict[str, Any]:
        """流式结束时构造 response_message（content/tool_calls，含推理内容时附加 reasoning_content）"""
        response_message = {
            "content": self.stream_content_buffer,
            "tool_calls": self.stream_tool_buffers
        }
        if self.stream_reasoning_buffer:
            response_message["reasoning_content"] = self.stream_reasoning_buffer
        return response_message

    @abc.abstractmethod
    def serialize_tool_description(self, tools_configs: Dict[str, Dict[str, Any]]) -> List[Dict]:
        """
        转换工具配置格式为Provider格式。配置格式：

        [tools.add_task]
        type = "function"
        description = "添加日程"
        param.datetime.type = "string"
        param.datetime.description = "日期时间，如 2025-12-31 12:34:56"
        param.datetime.required = false
        param.content.type = "string"
        param.content.description = "待办事项"
        param.content.required = true
        """
        pass

    @abc.abstractmethod
    def parse_tool_calls(self, tool_calls: List[Dict]) -> List[ToolCall]:
        """解析大模型返回的工具调用"""
        pass

    @abc.abstractmethod
    def serialize_message(self, message: Message) -> Dict[str, Any]:
        """将Message转为对应提供商的消息体"""
        pass

    @abc.abstractmethod
    def make_request_body(self, request: LLMProviderRequest) -> Dict[str, Any]:
        """构建符合Provider要求的请求体"""
        pass

    @classmethod
    def parse_response_message(cls, response: LLMProviderResponse, message: Dict[str, Any]) -> Message:
        return Message.from_content(message)

    @abc.abstractmethod
    def parse_batch_response(self, response: LLMProviderResponse, response_data: Dict[str, Any]) -> None:
        """解析非流式响应，填充LLMResponse"""
        pass

    @abc.abstractmethod
    def parse_stream_event(self, response: LLMProviderResponse, event: sse.ServerSentEvent) -> ProviderStreamEvent:
        """
        解析流式响应的单个chunk
        返回值：(是否结束, 内容片段, 推理片段, 工具调用列表)
        """
        pass
