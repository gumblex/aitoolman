import enum
import typing
import base64
import asyncio
import inspect
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Union, Callable

from .channel import TextFragmentChannel


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


class LLMApplicationError(LLMError):
    """Application code error"""
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


class MediaContent(typing.NamedTuple):
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

    def __repr__(self):
        result = 'MediaContent(%r' % self.media_type
        if self.raw_value:
            result += ', raw_value=...)'
        elif self.mime_type:
            result += ', mime_type=%r)' % self.mime_type
        elif self.filename:
            result += ', filename=%r)' % self.filename
        elif self.url:
            result += ', url=%r)' % self.url
        else:
            result += ')'
        return result

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


class Message(typing.NamedTuple):
    """给LLM发送的消息"""
    role: Optional[str] = None
    content: Optional[str] = None
    media_content: Optional[MediaContent] = None
    reasoning_content: Optional[str] = None
    tool_call_id: Optional[str] = None
    # 跟提供商有关的原始值，忽略所有上述字段
    raw_value: Optional[Dict] = None

    @classmethod
    def from_content(
            cls,
            content: Union[str, Dict],
            role: Optional[str] = "user",
            media_content: Optional[MediaContent] = None
    ):
        if isinstance(content, dict):
            return cls(raw_value=content)
        else:
            return cls(
                role=role, content=content,
                media_content=media_content
            )

    def to_dict(self) -> Dict[str, Any]:
        """将Message对象序列化为字典"""
        # 如果存在raw_value，直接返回其副本
        if self.raw_value is not None:
            return {"raw_value": self.raw_value}
        d: Dict[str, Any] = {
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
            return cls(content=data["raw_value"])

        return cls(
            role=data.get("role"),
            content=data.get("content"),
            reasoning_content=data.get("reasoning_content"),
            tool_call_id=data.get("tool_call_id"),
            raw_value=None,
            media_content=(
                MediaContent.from_dict(data["media_content"])
                if "media_content" in data else None
            )
        )


class ToolCall(typing.NamedTuple):
    """LLM回复的工具调用请求"""
    name: str
    arguments_text: str
    arguments: Optional[Dict[str, Any]]
    id: Optional[str] = None
    type: str = 'function'

    def __str__(self):
        return '%s(%s)' % (
            self.name,
            ', '.join('%s=%r' % (k, v) for k, v in self.arguments.items()) if self.arguments else self.arguments_text
        )


@dataclass
class LLMProviderResponse:
    """LLM网络层返回类，用于包装模型提供商的应答"""
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
class LLMProviderRequest:
    """LLM网络层请求类，用于包装向模型提供商发送的请求"""
    client_id: str
    context_id: Optional[str]
    request_id: str
    model_name: str
    messages: List[Message]
    tools: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    options: Dict[str, Any] = field(default_factory=dict)
    stream: bool = False
    output_channel: Optional[TextFragmentChannel] = field(default_factory=TextFragmentChannel)
    reasoning_channel: Optional[TextFragmentChannel] = None
    is_cancelled: bool = False
    response: asyncio.Future[LLMProviderResponse] = field(default_factory=asyncio.Future)


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
    # 应用程序代码错误
    error_app = "error: application"
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
        elif finish_reason_enum == FinishReason.error_app:
            raise LLMApplicationError(error_text or "Application error")
        elif finish_reason_enum == FinishReason.cancelled:
            raise LLMCancelledError(error_text or "Request cancelled")
        elif finish_reason_enum == FinishReason.unknown:
            raise LLMUnknownError(error_text or "Unknown error")
        elif finish_reason_enum == FinishReason.error:
            raise GenericError(error_text or "Generic error")
        else:
            raise LLMUnknownError(f"Unrecognized finish reason: {finish_reason}")


class LLMDirectRequest(typing.NamedTuple):
    """应用层实际请求参数"""
    model_name: str
    messages: List[Message]
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: bool = False
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None


class LLMModuleRequest(typing.NamedTuple):
    """应用层模板请求参数（模块配置）"""
    module_name: str
    template_params: Dict[str, Any]
    model_name: Optional[str] = None
    context_messages: List[Message] = []
    media_content: Optional[MediaContent] = None

    # 覆盖原始配置
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    options: Optional[Dict[str, Any]] = None
    stream: Optional[bool] = None
    output_channel: Union[str, TextFragmentChannel, None] = None
    reasoning_channel: Union[str, TextFragmentChannel, None] = None


@dataclass
class LLMModuleResult:
    """应用层（模板）请求响应"""
    module_name: str
    # 原始请求参数
    request: LLMDirectRequest = None
    # 原始响应
    response_text: str = ""
    response_reasoning: str = ""
    # 处理后的结果
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    # 状态信息
    status: FinishReason = FinishReason.stop
    error_text: Optional[str] = None
    # 原始模板参数
    request_params: Dict[str, Any] = field(default_factory=dict)
    # 原始响应（用于拼接上下文）
    response_message: Optional[Message] = None
    # 后处理结果
    data: Any = None

    @classmethod
    def from_response(cls, request: LLMDirectRequest, response: LLMProviderResponse) -> "LLMModuleResult":
        """从 LLMProviderResponse 转换为 LLMModuleResult"""
        return cls(
            module_name='',
            request=request,
            # 原始响应字段直接映射
            response_text=response.response_text,
            response_reasoning=response.response_reasoning,
            # 初始处理后的文本暂等同于原始响应（后续可按需重写）
            text=response.response_text,
            # 将工具调用列表转换为 工具名→ToolCall 的字典
            tool_calls=response.response_tool_calls,
            # 转换 finish_reason 为 FinishReason 枚举（None 时使用默认值 stop）
            status=(
                FinishReason(response.finish_reason)
                if response.finish_reason is not None else FinishReason.stop
            ),
            # 错误信息直接映射
            error_text=response.error_text,
            # 原始响应消息直接映射
            response_message=(
                Message.from_content(response.response_message)
                if response.response_message else None
            )
        )

    def raise_for_status(self):
        FinishReason.raise_for_status(self.status.value, self.error_text)

    async def run_tool_calls(self, fn_map: Dict[str, Callable]) -> List[Message]:
        """Execute tool calls using the provided function map.

        Args:
            self: LLMModuleResult object
            fn_map: Dictionary mapping tool names to callable functions

        Returns:
            Context messages for next request.

        Raises:
            LLMError: raise_for_status()
            LLMResponseFormatError: tool not found
        """
        self.raise_for_status()
        context = list(self.request.messages)
        context.append(self.response_message)
        if not self.tool_calls:
            return context

        for tool_call in self.tool_calls:
            tool_name = tool_call.name
            if tool_name not in fn_map:
                raise LLMResponseFormatError(f"Tool '{tool_name}' not found in function map.")

            func = fn_map[tool_name]
            # Call function with arguments if present
            if tool_call.arguments:
                result = func(**tool_call.arguments)
            else:
                result = func()
            if inspect.isawaitable(result):
                result = await result
            context.append(Message.from_content(content=str(result), role='tool'))
        return context
