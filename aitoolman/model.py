import enum
import typing
import base64
import asyncio
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

from .channel import TextChannel


class LLMError(RuntimeError):
    pass


class LLMLengthLimitError(LLMError):
    """Error when response reaches length limit"""
    pass


class LLMContentFilterError(LLMError):
    """Error when content is filtered"""
    pass


class LLMApiRequestError(LLMError):
    """Error with the request"""
    pass


class LLMResponseFormatError(LLMError):
    """Error with response format"""
    pass


class LLMCancelledError(LLMError):
    """Error when request is cancelled"""
    pass


class LLMUnknownError(LLMError):
    """Error for unknown finish reasons"""
    pass


class GenericError(LLMError):
    """Generic error"""
    pass


@dataclass
class MediaContent:
    """多媒体内容"""
    # image/video
    media_type: str
    # 按以下优先顺序
    # 1. raw_value
    raw_value: Optional[Dict] = None
    # 2. data+mime_type
    data: Optional[bytes] = None
    mime_type: Optional[str] = None
    # 3. filename
    filename: Optional[str] = None
    # 4. url
    url: Optional[str] = None
    options: Optional[Dict] = None

    def to_dict(self) -> Dict[str, Any]:
        """将MediaContent对象序列化为字典"""
        result = {
            "raw_value": self.raw_value,
            "media_type": self.media_type,
            "mime_type": self.mime_type,
            "filename": self.filename,
            "url": self.url,
            "options": self.options
        }
        if self.data is not None:
            result["data"] = base64.b64encode(self.data).decode('utf-8')
        return {k: v for k, v in result.items() if v is not None}

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "MediaContent":
        """从字典反序列化为MediaContent对象"""
        bytes_data = None
        if "data" in data and isinstance(data["data"], str):
            bytes_data = base64.b64decode(data["data"])

        # 创建MediaContent对象
        return cls(
            raw_value=data.get("raw_value"),
            media_type=data.get("media_type"),
            data=bytes_data,
            mime_type=data.get("mime_type"),
            filename=data.get("filename"),
            url=data.get("url"),
            options=data.get("options")
        )


@dataclass
class Message:
    """给LLM发送的消息"""
    role: Optional[str] = None
    content: Optional[str] = None
    media_content: Optional[MediaContent] = None
    reasoning_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    # 跟提供商有关的原始值，忽略所有上述字段
    raw_value: Optional[Dict] = None

    def __init__(
            self,
            content: Union[str, Dict],
            role: Optional[str] = "user",
            media_content: Optional[MediaContent] = None
    ):
        if isinstance(content, dict):
            self.raw_value = content
        else:
            self.role = role
            self.content = content
            self.media_content = media_content

    def to_dict(self) -> Dict[str, Any]:
        """将Message对象序列化为字典"""
        # 如果存在raw_value，直接返回其副本
        if self.raw_value is not None:
            return {"raw_value": self.raw_value}
        d = {
            "role": self.role,
            "content": self.content,
            "reasoning_content": self.reasoning_content,
            "tool_call_id": self.tool_call_id
        }
        if self.media_content:
            d["media_content"] = self.media_content.to_dict()
        return d

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Message":
        """从字典反序列化为Message对象"""
        if "raw_value" in data:
            # 如果有显式的raw_value字段，直接使用
            return cls(content=data["raw_value"], role=None)

        message = cls.__new__(cls)
        message.role = data.get("role")
        message.content = data.get("content")
        message.reasoning_content = data.get("reasoning_content")
        message.tool_call_id = data.get("tool_call_id")
        message.raw_value = None
        if "media_content" in data:
            message.media_content = MediaContent.from_dict(
                data["media_content"])
        else:
            message.media_content = None

        return message


@dataclass
class ToolCall:
    """LLM回复的工具调用请求"""
    name: str
    arguments_text: str
    arguments: Optional[Dict[str, Any]]
    id: Optional[str] = None
    type: str = 'function'


@dataclass
class LLMResponse:
    client_id: str
    context_id: str
    request_id: str
    model_name: str
    stream: bool

    # 响应时间
    start_time: Optional[float] = None
    queue_time: Optional[float] = None
    queue_length: Optional[int] = None
    # 第一条响应时间
    time_to_first_token: Optional[float] = None
    # http 发送-完整响应时间
    total_response_time: Optional[float] = None

    # 响应内容
    response_text: str = ""
    response_reasoning: str = ""
    response_tool_calls: List[ToolCall] = field(default_factory=list)

    # 完成信息
    finish_reason: Optional[str] = None
    error_text: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None

    # 完整请求/响应数据
    response_message: Optional[Dict[str, Any]] = None

    def raise_for_status(self):
        FinishReason.raise_for_status(self.finish_reason, self.error_text)


@dataclass
class LLMRequest:
    """LLM请求类"""
    client_id: str
    context_id: Optional[str]
    request_id: str
    model_name: str
    messages: List[Message]
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    response_channel: Optional[TextChannel] = None
    reasoning_channel: Optional[TextChannel] = None
    is_cancelled: bool = False
    response: asyncio.Future[LLMResponse] = field(default_factory=asyncio.Future)

    def __post_init__(self):
        if self.response_channel is None:
            self.response_channel = TextChannel(read_fragments=self.stream)


class FinishReason(enum.Enum):
    # 提供商原因
    # 模型输出自然结束
    stop = "stop"
    # 长度限制
    length = "length"
    # 内容审核
    content_filter = "content_filter"
    # 调用了工具
    tool_calls = "tool_calls"

    # 本地原因
    # 通用错误
    error = "error"
    # 请求错误
    error_request = "error: request"
    # 返回格式错误
    error_format = "error: format"
    # 取消
    cancelled = "cancelled"

    # 未知：不应该出现
    unknown = "unknown"

    @staticmethod
    def raise_for_status(finish_reason: str, error_text: str = ""):
        """Raise appropriate error if the response indicates failure"""
        if not finish_reason:
            return

        # Get the enum value for comparison
        try:
            finish_reason_enum = FinishReason(finish_reason)
        except ValueError:
            raise LLMUnknownError(f"Unrecognized finish reason: {finish_reason}")

        if finish_reason_enum == FinishReason.stop:
            return
        elif finish_reason_enum == FinishReason.tool_calls:
            return
        elif finish_reason_enum == FinishReason.length:
            raise LLMLengthLimitError(error_text or "Response reached length limit")
        elif finish_reason_enum == FinishReason.content_filter:
            raise LLMContentFilterError(error_text or "Content filtered")
        elif finish_reason_enum == FinishReason.error_request:
            raise LLMApiRequestError(error_text or "Request error")
        elif finish_reason_enum == FinishReason.error_format:
            raise LLMResponseFormatError(error_text or "Format error")
        elif finish_reason_enum == FinishReason.cancelled:
            raise LLMCancelledError(error_text or "Request cancelled")
        elif finish_reason_enum == FinishReason.unknown:
            raise LLMUnknownError(error_text or "Unknown error")
        elif finish_reason_enum == FinishReason.error:
            raise GenericError(error_text or "Generic error")
        else:
            raise LLMUnknownError(f"Unrecognized finish reason: {finish_reason}")


@dataclass
class LLMModuleResult:
    # 原始响应
    response_text: str = ""
    response_reasoning: str = ""
    # 处理后的结果
    text: str = ""
    # name -> ToolCall
    tool_calls: Dict[str, ToolCall] = field(default_factory=dict)
    # 状态信息
    status: FinishReason = FinishReason.stop
    error_text: Optional[str] = None
    # 原始请求参数
    request_params: Dict[str, Any] = field(default_factory=dict)
    # 原始请求和响应（用于拼接上下文）
    request_messages: List[Message] = field(default_factory=list)
    response_message: Optional[Dict[str, Any]] = None
    # 后处理结果
    data: Any = None

    @classmethod
    def from_response(cls, response: LLMResponse) -> "LLMModuleResult":
        """从 LLMResponse 转换为 LLMModuleResult"""
        return cls(
            # 原始响应字段直接映射
            response_text=response.response_text,
            response_reasoning=response.response_reasoning,
            # 初始处理后的文本暂等同于原始响应（后续可按需重写）
            text=response.response_text,
            # 将工具调用列表转换为 工具名→ToolCall 的字典
            tool_calls={call.name: call for call in response.response_tool_calls},
            # 转换 finish_reason 为 FinishReason 枚举（None 时使用默认值 stop）
            status=FinishReason(response.finish_reason) if response.finish_reason is not None else FinishReason.stop,
            # 错误信息直接映射
            error_text=response.error_text,
            # 原始响应消息直接映射
            response_message=response.response_message
        )

    def raise_for_status(self):
        FinishReason.raise_for_status(self.status.value, self.error_text)

    def call(self, fn_map: Dict[str, Callable]) -> Dict[str, Any]:
        """Execute tool calls using the provided function map.

        Args:
            fn_map: Dictionary mapping tool names to callable functions

        Returns:
            Dictionary with tool call IDs as keys and function results as values

        Raises:
            LLMError: raise_for_status()
            LLMResponseFormatError: tool not found
        """
        self.raise_for_status()
        if not self.tool_calls:
            return {}

        results = {}
        for tool_call in self.tool_calls.values():
            tool_name = tool_call.name
            if tool_name not in fn_map:
                raise LLMResponseFormatError(f"Tool '{tool_name}' not found in function map.")

            func = fn_map[tool_name]
            # Call function with arguments if present
            if tool_call.arguments:
                result = func(**tool_call.arguments)
            else:
                result = func()
            results[tool_call.id] = result

        return results

