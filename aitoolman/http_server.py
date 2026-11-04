"""OpenAI 兼容 HTTP 服务。

将 aitoolman 的配置管理、多模型选择与请求能力包装为标准的 OpenAI
chat/completions API。支持两种后端模式，HTTP 层代码路径完全一致：

- 直连模式（默认，未配置 [http].zmq_endpoint）：通过配置的提供商详情
  直接提供服务。
- 远程模式（配置 [http].zmq_endpoint）：连接已有 ZMQ 服务。

配置来源 llm_provider.toml 的 [http] 节，CLI 参数可单项覆盖（CLI > 配置 >
内置默认）。
"""
import asyncio
import logging
import secrets
import sys
import threading
import time
from typing import Any, Callable, Dict, List, Optional

import msgspec
import zmq
import zmq.asyncio

try:
    from aiohttp import web
except ImportError:
    web = None  # 缺失时由 build_http_config/serve 统一报错

from . import util
from .channel import Channel
from .model import (
    LLMContentFilterError,
    LLMLengthLimitError,
    LLMNoAvailableModelError,
    MediaContent,
    Message,
)
from .zmqclient import LLMZmqClient
from .zmqserver import LLMZmqServer

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# OpenAI 协议数据结构
# ---------------------------------------------------------------------------

class ErrorResponse(msgspec.Struct):
    """错误详情，作为 {"error": {...}} 信封的内层对象"""
    message: str
    type: str
    code: Optional[str] = None


class ErrorEnvelope(msgspec.Struct):
    """统一错误信封 {"error":{"message","type","code"}}"""
    error: ErrorResponse


class Usage(msgspec.Struct):
    """OpenAI token 用量统计"""
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0


class ResponseMessage(msgspec.Struct):
    """chat.completion 响应中的 message 对象"""
    role: str = "assistant"
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class Choice(msgspec.Struct):
    """chat.completion 响应中的 choice 对象"""
    index: int
    message: ResponseMessage
    finish_reason: Optional[str] = None


class ChatCompletion(msgspec.Struct):
    """POST /v1/chat/completions 阻塞响应"""
    id: str
    object: str
    created: int
    model: str
    choices: List[Choice]
    usage: Usage


class Delta(msgspec.Struct, omit_defaults=True):
    """流式 chunk 中的增量内容"""
    role: Optional[str] = None
    content: Optional[str] = None
    reasoning_content: Optional[str] = None
    tool_calls: Optional[List[Dict[str, Any]]] = None


class ChunkChoice(msgspec.Struct, omit_defaults=True):
    """流式 chunk 中的 choice 对象"""
    index: int
    delta: Delta
    finish_reason: Optional[str] = None


class ChatCompletionChunk(msgspec.Struct, omit_defaults=True):
    """POST /v1/chat/completions 流式响应 chunk"""
    id: str
    object: str
    created: int
    model: str
    choices: List[ChunkChoice]
    usage: Optional[Usage] = None


# ---------------------------------------------------------------------------
# [http] 配置装配
# ---------------------------------------------------------------------------

DEFAULT_HTTP_HOST = "127.0.0.1"
DEFAULT_HTTP_PORT = 8000
# 无需 Bearer 认证的路径白名单
AUTH_WHITELIST_PATHS = ("/", "/health")


class HttpServerConfig(msgspec.Struct):
    """HTTP 服务运行配置（[http] 节 + CLI 覆盖结果）"""
    host: str
    port: int
    zmq_endpoint: Optional[str]      # None=直连模式；否则为远程 ZMQ 服务端点
    zmq_auth_token: Optional[str]    # 远程模式认证 token
    client_tokens: List[str]         # 客户端 Bearer 令牌列表

    @property
    def mode(self) -> str:
        """后端模式：direct（直连）/ remote（远程 ZMQ 服务）"""
        return "remote" if self.zmq_endpoint else "direct"


# app 存储键：aiohttp 可用时使用 web.AppKey（消除 NotAppKeyWarning）
if web is not None:
    APP_CONFIG_KEY = web.AppKey("config", HttpServerConfig)
    APP_CLIENT_KEY = web.AppKey("client", LLMZmqClient)
    APP_MOCK_HANDLER_KEY = web.AppKey("mock_handler", object)
else:
    APP_CONFIG_KEY = "config"
    APP_CLIENT_KEY = "client"
    APP_MOCK_HANDLER_KEY = "mock_handler"


def build_http_config(
        http_config: Dict[str, Any],
        overrides: Optional[Dict[str, Any]] = None) -> HttpServerConfig:
    """从 llm_provider.toml 的 [http] 节装配 HTTP 服务配置。

    覆盖规则：CLI 参数（overrides）> [http] 节 > 内置默认。
    overrides 支持键：host / port / zmq_endpoint / zmq_auth_token /
    client_tokens（追加语义，与配置列表合并）；值为 None 视为未提供。

    Raises:
        RuntimeError: aiohttp 未安装
        ValueError: client_tokens 为空（不允许匿名访问），
                    或远程模式缺少 zmq_auth_token
    """
    if web is None:
        raise RuntimeError(
            "aiohttp is required for the HTTP server, "
            "install it with: pip install 'aitoolman[http]'")
    overrides = overrides or {}

    host = overrides.get('host') or http_config.get('host') or DEFAULT_HTTP_HOST
    port = overrides.get('port') or http_config.get('port') or DEFAULT_HTTP_PORT
    zmq_endpoint = overrides.get('zmq_endpoint')
    if zmq_endpoint is None:
        zmq_endpoint = http_config.get('zmq_endpoint') or None
    zmq_auth_token = overrides.get('zmq_auth_token')
    if zmq_auth_token is None:
        zmq_auth_token = http_config.get('zmq_auth_token') or None

    client_tokens = [t for t in (http_config.get('client_tokens') or []) if t]
    for token in (overrides.get('client_tokens') or []):
        if token and token not in client_tokens:
            client_tokens.append(token)

    if not client_tokens:
        raise ValueError(
            "HTTP server requires client_tokens: set [http].client_tokens in "
            "the config file or pass --client-token; "
            "anonymous access is not allowed")
    if zmq_endpoint is not None and not zmq_auth_token:
        raise ValueError(
            "zmq_auth_token is required in remote mode "
            "([http].zmq_endpoint is configured)")

    return HttpServerConfig(
        host=host,
        port=int(port),
        zmq_endpoint=zmq_endpoint,
        zmq_auth_token=zmq_auth_token,
        client_tokens=client_tokens,
    )


# ---------------------------------------------------------------------------
# 响应构造辅助
# ---------------------------------------------------------------------------

def make_json_response(struct: msgspec.Struct, status: int = 200,
                       headers: Optional[Dict[str, str]] = None) -> "web.Response":
    """将 msgspec.Struct 序列化为 JSON 响应"""
    return web.Response(
        body=msgspec.json.encode(struct),
        status=status,
        content_type="application/json",
        headers=headers,
    )


def make_error_response(message: str, error_type: str,
                        code: Optional[str] = None, status: int = 400,
                        headers: Optional[Dict[str, str]] = None) -> "web.Response":
    """构造 OpenAI 风格错误响应 {"error":{"message","type","code"}}"""
    envelope = ErrorEnvelope(error=ErrorResponse(
        message=message, type=error_type, code=code))
    return make_json_response(envelope, status=status, headers=headers)


# ---------------------------------------------------------------------------
# OpenAI 协议转换（请求侧纯函数）
# ---------------------------------------------------------------------------

# chat/completions 请求体中的协议保留键，其余键全部透传进 options
_RESERVED_BODY_KEYS = frozenset({"model", "messages", "stream", "tools"})
# 阻塞响应中视为正常完成的 finish_reason，其余走错误路径
_OK_FINISH_REASONS = frozenset({"stop", "length", "tool_calls", "content_filter"})


def parse_openai_messages(raw_messages: List[Any]) -> List[Message]:
    """将 OpenAI messages 数组转换为 aitoolman Message 列表。

    - content 为字符串：直接构造；role=tool 且带 tool_call_id 时构造对应字段
    - content 为数组：text part 拼接为文本、image_url part 转为
      MediaContent(image, url=...)；含未知 part 类型时整条消息以 raw_value
      原样透传
    - assistant 消息带 tool_calls：以 raw_value 原样透传
    - 其余形态（content 缺失等）：以 raw_value 原样透传

    Raises:
        ValueError: 消息不是对象或缺少字符串 role
    """
    result: List[Message] = []
    for raw in raw_messages:
        if not isinstance(raw, dict):
            raise ValueError("each message must be an object")
        role = raw.get("role")
        if not isinstance(role, str) or not role:
            raise ValueError("each message requires a string 'role'")
        if raw.get("tool_calls"):
            # assistant 工具调用消息：保留 OpenAI 原始结构无损透传
            result.append(Message(raw_value=raw))
            continue
        content = raw.get("content")
        if isinstance(content, str):
            result.append(Message(
                role=role,
                content=content,
                tool_call_id=raw.get("tool_call_id") if role == "tool" else None,
            ))
            continue
        if isinstance(content, list):
            texts: List[str] = []
            media: List[MediaContent] = []
            has_unknown = False
            for part in content:
                if not isinstance(part, dict):
                    has_unknown = True
                    break
                part_type = part.get("type")
                if part_type == "text":
                    texts.append(str(part.get("text") or ""))
                elif part_type == "image_url":
                    image_url = part.get("image_url")
                    url = (image_url.get("url")
                           if isinstance(image_url, dict) else None)
                    if not url:
                        has_unknown = True
                        break
                    media.append(MediaContent(media_type="image", url=url))
                else:
                    has_unknown = True
                    break
            if has_unknown:
                result.append(Message(raw_value=raw))
            else:
                result.append(Message(
                    role=role,
                    content="".join(texts),
                    media_content=media or None,
                ))
            continue
        # content 缺失/为 None 等其他形态：原样透传
        result.append(Message(raw_value=raw))
    return result


def parse_openai_tools(raw_tools: List[Any]) -> Dict[str, Dict[str, Any]]:
    """将 OpenAI tools 数组转换为 aitoolman 工具配置格式。

    [{type: "function", function: {name, description, parameters}}] 转为
    {name: {type: "function", description, param: {<名>: {type, description, required?}}}}，
    parameters.properties 映射到 param，required 数组标记 required。

    Raises:
        ValueError: 缺少 function/name 或 parameters 结构非法
    """
    result: Dict[str, Dict[str, Any]] = {}
    for tool in raw_tools:
        if (not isinstance(tool, dict) or tool.get("type") != "function"
                or not isinstance(tool.get("function"), dict)):
            raise ValueError(
                "each tool must be {type: 'function', function: {...}}")
        function = tool["function"]
        name = function.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("tool function requires a string 'name'")
        parameters = function.get("parameters") or {}
        if not isinstance(parameters, dict):
            raise ValueError("tool 'parameters' must be an object")
        properties = parameters.get("properties") or {}
        required = parameters.get("required") or []
        if not isinstance(properties, dict) or not isinstance(required, list):
            raise ValueError("tool 'parameters' must be a JSON schema object")
        param: Dict[str, Dict[str, Any]] = {}
        for prop_name, prop_schema in properties.items():
            schema = prop_schema if isinstance(prop_schema, dict) else {}
            entry: Dict[str, Any] = {
                "type": schema.get("type", "string"),
                "description": schema.get("description", ""),
            }
            if prop_name in required:
                entry["required"] = True
            param[prop_name] = entry
        result[name] = {
            "type": "function",
            "description": function.get("description", ""),
            "param": param,
        }
    return result


# ---------------------------------------------------------------------------
# OpenAI 协议转换（响应侧）
# ---------------------------------------------------------------------------

def _tool_calls_to_openai(tool_calls: List[Any]) -> List[Dict[str, Any]]:
    """将 aitoolman ToolCall 列表转回 OpenAI tool_calls 数组格式"""
    return [
        {
            "id": tc.id,
            "type": tc.type,
            "function": {"name": tc.name, "arguments": tc.arguments_text},
        }
        for tc in tool_calls
    ]


def build_chat_completion_response(response) -> ChatCompletion:
    """从 LLMProviderResponse 构造阻塞 ChatCompletion 响应"""
    tool_calls = (
        _tool_calls_to_openai(response.response_tool_calls)
        if response.response_tool_calls else None)
    prompt_tokens = response.prompt_tokens or 0
    completion_tokens = response.completion_tokens or 0
    return ChatCompletion(
        id="chatcmpl-" + response.request_id,
        object="chat.completion",
        created=int(time.time()),
        model=response.model_name,
        choices=[Choice(
            index=0,
            message=ResponseMessage(
                role="assistant",
                content=response.response_text or None,
                reasoning_content=response.response_reasoning or None,
                tool_calls=tool_calls,
            ),
            finish_reason=response.finish_reason,
        )],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )


def error_envelope_for_finish_reason(
        finish_reason: Optional[str],
        error_text: Optional[str]) -> Optional[ErrorEnvelope]:
    """finish_reason 为错误态时构造错误信封；正常完成返回 None。

    error:request（上游请求失败）映射 upstream_error，其余错误态
    （error:format / error:application / cancelled / unknown 等）映射
    internal_error。
    """
    if not finish_reason or finish_reason in _OK_FINISH_REASONS:
        return None
    if finish_reason == "error: request":
        return ErrorEnvelope(error=ErrorResponse(
            message=error_text or "Upstream provider request failed",
            type="upstream_error"))
    return ErrorEnvelope(error=ErrorResponse(
        message=error_text or ("Request failed: %s" % finish_reason),
        type="internal_error"))


def error_response_for_finish_reason(
        finish_reason: Optional[str],
        error_text: Optional[str]) -> Optional["web.Response"]:
    """finish_reason 为错误态时构造错误响应；正常完成返回 None。

    upstream_error → 502，其余错误态 → 500。
    """
    envelope = error_envelope_for_finish_reason(finish_reason, error_text)
    if envelope is None:
        return None
    status = 502 if envelope.error.type == "upstream_error" else 500
    return make_json_response(envelope, status=status)


# ---------------------------------------------------------------------------
# SSE 流式帧
# ---------------------------------------------------------------------------

# 流式响应终止帧
SSE_DONE = b"data: [DONE]\n\n"


def sse_frame(payload: Any) -> bytes:
    """将 payload 编码为一帧 SSE 数据：data: {json}\\n\\n"""
    return b"data: " + msgspec.json.encode(payload) + b"\n\n"


def build_chat_completion_chunk(
        chunk_id: str, created: int, model: str,
        delta: Optional[Delta] = None,
        finish_reason: Optional[str] = None) -> ChatCompletionChunk:
    """构造流式 chunk（delta 增量帧或 finish 结束帧）"""
    return ChatCompletionChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[ChunkChoice(
            index=0,
            delta=delta if delta is not None else Delta(),
            finish_reason=finish_reason)],
    )


def build_usage_chunk(chunk_id: str, created: int, model: str,
                      prompt_tokens: int,
                      completion_tokens: int) -> ChatCompletionChunk:
    """构造 stream_options.include_usage 的最终 usage chunk（choices 为空）"""
    return ChatCompletionChunk(
        id=chunk_id,
        object="chat.completion.chunk",
        created=created,
        model=model,
        choices=[],
        usage=Usage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            total_tokens=prompt_tokens + completion_tokens,
        ),
    )





# ---------------------------------------------------------------------------
# 业务端点 handler
# ---------------------------------------------------------------------------

async def handle_chat_completions(
        request: "web.Request") -> "web.StreamResponse":
    """POST /v1/chat/completions（stream=false 阻塞 / stream=true SSE 流式）。

    协议保留键 model/messages/stream/tools 先提取，其余 body 键全量透传进
    options（provider 层按 {**body_options, **options} 合并，保证请求 body
    优先于配置 body_options）。stream=true 时委托 _stream_chat_completions。
    已注册 mock_handler 钩子时在 resolve_model 之前拦截请求。
    """
    client = request.app[APP_CLIENT_KEY]
    try:
        body = util.decode_json(await request.read())
    except Exception:
        return make_error_response(
            "Invalid JSON in request body", "invalid_request_error",
            code="invalid_json", status=400)
    if not isinstance(body, dict):
        return make_error_response(
            "Request body must be a JSON object", "invalid_request_error",
            status=400)

    model = body.get("model")
    if not isinstance(model, str) or not model:
        return make_error_response(
            "'model' is a required property", "invalid_request_error",
            status=400)

    mock_handler = request.app[APP_MOCK_HANDLER_KEY]
    if mock_handler is not None:
        # 可选请求拦截钩子：认证之后、resolve_model 之前调用；
        # 返回响应对象则直接返回，None 放行进入正常业务路径
        intercepted = await mock_handler(request, body)
        if intercepted is not None:
            return intercepted

    raw_messages = body.get("messages")
    if not isinstance(raw_messages, list) or not raw_messages:
        return make_error_response(
            "'messages' must be a non-empty array", "invalid_request_error",
            status=400)
    try:
        messages = parse_openai_messages(raw_messages)
    except ValueError as e:
        return make_error_response(str(e), "invalid_request_error", status=400)

    raw_tools = body.get("tools")
    tools: Optional[Dict[str, Dict[str, Any]]] = None
    if raw_tools is not None:
        if not isinstance(raw_tools, list):
            return make_error_response(
                "'tools' must be an array", "invalid_request_error",
                status=400)
        try:
            tools = parse_openai_tools(raw_tools)
        except ValueError as e:
            return make_error_response(
                str(e), "invalid_request_error", status=400)

    options = {k: v for k, v in body.items() if k not in _RESERVED_BODY_KEYS}

    # 模型解析先于流式响应 prepare，失败时可直接返回普通 HTTP 错误
    try:
        candidates = await client.resolve_model(model, messages=messages)
        actual_model = candidates[0]
    except LLMNoAvailableModelError as e:
        return make_error_response(str(e), "model_not_found", status=404)
    except LLMLengthLimitError as e:
        return make_error_response(
            str(e), "context_length_exceeded", status=400)
    except LLMContentFilterError as e:
        return make_error_response(str(e), "content_filter", status=400)
    except Exception:
        logger.exception("chat/completions resolve model error")
        return make_error_response(
            "Internal server error", "internal_error", status=500)

    if body.get("stream"):
        return await _stream_chat_completions(
            request, client, actual_model, messages, tools or {}, options)

    try:
        provider_request = await client.request(
            model_name=actual_model,
            messages=messages,
            tools=tools or {},
            options=options,
            stream=False,
        )
        response = await provider_request.response
    except Exception:
        logger.exception("chat/completions internal error")
        return make_error_response(
            "Internal server error", "internal_error", status=500)

    error_resp = error_response_for_finish_reason(
        response.finish_reason, response.error_text)
    if error_resp is not None:
        return error_resp
    return make_json_response(
        build_chat_completion_response(response),
        headers={"x-aitoolman-resolved-model": actual_model})


async def _cancel_upstream(client: LLMZmqClient, provider_request) -> None:
    """客户端断连后取消上游请求（清理失败仅记录，不掩盖原异常）"""
    try:
        await client.cancel(provider_request.request_id)
    except Exception:
        logger.debug("cancel upstream request %s failed",
                     provider_request.request_id, exc_info=True)


async def _stream_chat_completions(
        request: "web.Request", client: LLMZmqClient, actual_model: str,
        messages: List[Message], tools: Dict[str, Dict[str, Any]],
        options: Dict[str, Any]) -> "web.StreamResponse":
    """stream=true 流式路径：channel 事件泵为 SSE delta 帧。

    response topic → delta.content、reasoning topic → delta.reasoning_content；
    通道结束（EOFError）后取最终 LLMProviderResponse，发 finish_reason chunk
    （include_usage 时再发空 choices 的 usage chunk），最后以 data: [DONE] 终止；
    最终 finish_reason 为错误态时改发一帧错误信封、仍补发 [DONE]；
    客户端断连（CancelledError/ConnectionError）时取消上游请求。
    """
    stream_options = options.get("stream_options")
    include_usage = bool(
        isinstance(stream_options, dict)
        and stream_options.get("include_usage"))

    try:
        provider_request = await client.request(
            model_name=actual_model,
            messages=messages,
            tools=tools,
            options=options,
            stream=True,
            output_channel=Channel(),
        )
    except Exception:
        # prepare 之前失败仍可返回普通错误响应
        logger.exception("chat/completions failed to start stream")
        return make_error_response(
            "Internal server error", "internal_error", status=500)

    chunk_id = "chatcmpl-" + provider_request.request_id
    created = int(time.time())

    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
        "x-aitoolman-resolved-model": actual_model,
    })
    await resp.prepare(request)

    try:
        while True:
            # 传输层已关闭（客户端断连）：统一转为 ConnectionResetError 处理
            transport = request.transport
            if transport is None or transport.is_closing():
                raise ConnectionResetError("client disconnected")
            try:
                event = await provider_request.output_channel.read()
            except EOFError:
                break
            if not isinstance(event.data, str):
                # data=None 为该 topic 的结束事件；response 通道结束即流结束
                # （ZMQ 链路不传播 EOF，provider 收尾写入 (response, None)）
                if event.topic == "response":
                    break
                continue
            if event.topic == "response":
                delta = Delta(content=event.data)
            elif event.topic == "reasoning":
                delta = Delta(reasoning_content=event.data)
            else:
                continue
            await resp.write(sse_frame(
                build_chat_completion_chunk(chunk_id, created, actual_model,
                                            delta=delta)))

        response = await provider_request.response
        error_envelope = error_envelope_for_finish_reason(
            response.finish_reason, response.error_text)
        if error_envelope is not None:
            await resp.write(sse_frame(error_envelope))
        else:
            await resp.write(sse_frame(build_chat_completion_chunk(
                chunk_id, created, response.model_name,
                finish_reason=response.finish_reason)))
            if include_usage:
                await resp.write(sse_frame(build_usage_chunk(
                    chunk_id, created, response.model_name,
                    response.prompt_tokens or 0,
                    response.completion_tokens or 0)))
        await resp.write(SSE_DONE)
        await resp.write_eof()
    except asyncio.CancelledError:
        # 客户端断连（handler 被取消）：取消上游请求后向上传播
        await _cancel_upstream(client, provider_request)
        raise
    except ConnectionError:
        # 客户端断连：取消上游请求，按连接中断结束
        await _cancel_upstream(client, provider_request)
    except Exception:
        await _cancel_upstream(client, provider_request)
        raise
    return resp


async def handle_list_models(request: "web.Request") -> "web.Response":
    """GET /v1/models：列出可用模型，标签作为模型条目一并返回"""
    client = request.app[APP_CLIENT_KEY]
    try:
        models = await client.list_models()
        tags = await client.list_tags()
    except Exception:
        logger.exception("list_models/list_tags failed")
        return make_error_response(
            "Internal server error", "internal_error", status=500)
    # 先构造模型名与标签名列表，再统一构造模型条目输出；
    # 与模型重名的标签跳过（模型优先），空标签不输出
    model_names = [model.name for model in models]
    tag_names = [
        tag for tag, tag_models in tags.items()
        if tag_models and tag not in model_names
    ]
    data = [
        {
            "id": name,
            "object": "model",
            "created": 0,
            "owned_by": "aitoolman",
        }
        for name in model_names + tag_names
    ]
    return web.json_response({"object": "list", "data": data})


# ---------------------------------------------------------------------------
# 应用工厂与端点
# ---------------------------------------------------------------------------

def create_app(config: HttpServerConfig,
               client: Optional[LLMZmqClient],
               mock_handler: Optional[Callable] = None) -> "web.Application":
    """创建 aiohttp 应用：Bearer 认证中间件 + 已注册端点。

    业务端点（chat/completions 等）后续注册；配置与 ZMQ 客户端经
    app['config'] / app['client'] 暴露给 handler。mock_handler 为可选的
    chat/completions 请求拦截钩子，经 APP_MOCK_HANDLER_KEY 存入 app。
    """

    @web.middleware
    async def auth_middleware(request: "web.Request", handler):
        """Bearer 认证：白名单路径放行，其余校验 client_tokens"""
        if request.path in AUTH_WHITELIST_PATHS:
            return await handler(request)
        auth_header = request.headers.get("Authorization", "")
        token = (auth_header[len("Bearer "):]
                 if auth_header.startswith("Bearer ") else "")
        if token:
            if token in config.client_tokens:
                return await handler(request)
        return make_error_response(
            "Invalid authentication credentials",
            "authentication_error",
            code="invalid_api_key",
            status=401,
            headers={"WWW-Authenticate": "Bearer"},
        )

    async def handle_status(request: "web.Request") -> "web.Response":
        return web.json_response({"status": "ok", "mode": config.mode})

    app = web.Application(middlewares=[auth_middleware])
    app[APP_CONFIG_KEY] = config
    app[APP_CLIENT_KEY] = client
    app[APP_MOCK_HANDLER_KEY] = mock_handler
    app.router.add_get("/", handle_status)
    app.router.add_get("/health", handle_status)
    app.router.add_post("/v1/chat/completions", handle_chat_completions)
    app.router.add_get("/v1/models", handle_list_models)
    return app


# ---------------------------------------------------------------------------
# 双模式生命周期
# ---------------------------------------------------------------------------

def _build_direct_server_config(provider_config: Dict[str, Any],
                                endpoint: str,
                                auth_token: str) -> Dict[str, Any]:
    """构造直连模式后台 LLMZmqServer 的配置。

    基于 provider_config 拷贝并覆盖 [server] 节：inproc 端点与内部随机认证
    token；不发布审计事件（PUB 不绑定，审计消息静默丢弃）。
    """
    server_config = dict(provider_config)
    server_section = dict(provider_config.get('server') or {})
    server_section['zmq_router_rpc'] = endpoint
    server_section['zmq_auth_token'] = auth_token
    server_section.pop('zmq_pub_event', None)
    server_section.pop('zmq_pub_event_type', None)
    server_config['server'] = server_section
    return server_config


async def _start_direct_backend(provider_config: Dict[str, Any], ctx,
                                stop: threading.Event,
                                ready_timeout: float = 10.0):
    """在后台线程启动 LLMZmqServer，返回 (LLMZmqClient, thread)。

    - inproc:// 随机端点 + secrets.token_hex(16) 内部认证 token
    - on_ready Event 同步保证服务端 bind 先于客户端 connect
    - 初始化失败经 server.start_error 快速传播
    """
    endpoint = "inproc://aitoolman-http-" + util.get_id()
    auth_token = secrets.token_hex(16)
    server = LLMZmqServer(
        _build_direct_server_config(provider_config, endpoint, auth_token),
        ctx=ctx)
    on_ready = threading.Event()
    thread = threading.Thread(
        target=asyncio.run,
        args=(server.run(on_ready, stop),),
        daemon=True,
        name="aitoolman-zmq-server",
    )
    thread.start()
    # 轮询等待就绪；初始化失败快速传播，不阻塞事件循环
    loop = asyncio.get_running_loop()
    deadline = loop.time() + ready_timeout
    while not on_ready.is_set():
        if server.start_error is not None:
            raise RuntimeError(
                "ZMQ server failed to start: %s" % server.start_error)
        if loop.time() > deadline:
            raise RuntimeError(
                "ZMQ server failed to start within %ss" % ready_timeout)
        await asyncio.sleep(0.05)
    client = LLMZmqClient(endpoint, auth_token, ctx=ctx)
    return client, thread


async def serve(config: HttpServerConfig, provider_config: Dict[str, Any],
                stop_event: Optional[threading.Event] = None,
                mock_handler: Optional[Callable] = None) -> None:
    """启动 HTTP 服务并阻塞直到 stop_event 置位。

    直连模式：后台线程运行 LLMZmqServer（与主线程共享 zmq.asyncio.Context，
    inproc 通信），主线程运行 aiohttp；远程模式：LLMZmqClient 直连已有 ZMQ
    服务。退出顺序：runner.cleanup → client.close → 服务端停止 →
    thread.join → ctx.term。

    Args:
        config: HTTP 服务配置（build_http_config 装配结果）
        provider_config: llm_provider.toml 完整配置（直连模式构造服务端用）
        stop_event: 可选停止信号；不传时由 KeyboardInterrupt 触发清理
        mock_handler: 可选的 chat/completions 请求拦截钩子，透传给 create_app
    """
    if web is None:
        raise RuntimeError(
            "aiohttp is required for the HTTP server, "
            "install it with: pip install 'aitoolman[http]'")
    stop = stop_event if stop_event is not None else threading.Event()
    ctx = zmq.asyncio.Context()
    client: Optional[LLMZmqClient] = None
    server_thread: Optional[threading.Thread] = None
    runner: Optional["web.AppRunner"] = None
    try:
        if config.zmq_endpoint is not None:
            client = LLMZmqClient(
                config.zmq_endpoint, config.zmq_auth_token, ctx=ctx)
        else:
            client, server_thread = await _start_direct_backend(
                provider_config, ctx, stop)
        await client.initialize()

        app = create_app(config, client, mock_handler=mock_handler)
        runner = web.AppRunner(app)
        await runner.setup()
        site = web.TCPSite(runner, config.host, config.port)
        await site.start()
        logger.info("HTTP server listening on http://%s:%d (mode=%s)",
                    config.host, config.port, config.mode)

        # 阻塞直到 stop_event 置位（CLI 模式由 KeyboardInterrupt 中断）
        while not stop.is_set():
            await asyncio.sleep(0.1)
    finally:
        if runner is not None:
            await runner.cleanup()
        if client is not None:
            await client.close()
        if server_thread is not None:
            # 服务端主循环轮询同一 stop_event，置位后自动退出
            stop.set()
            await asyncio.to_thread(server_thread.join, 5.0)
            if server_thread.is_alive():
                logger.warning("ZMQ server thread did not exit within 5s")
        await asyncio.to_thread(ctx.term)


# ---------------------------------------------------------------------------
# CLI 入口
# ---------------------------------------------------------------------------

def run_http(args) -> None:
    """http 子命令入口（供 cli.py 调用）：加载配置并启动 HTTP 服务。

    Args:
        args: argparse Namespace，读取属性 config / host / port /
              zmq_endpoint / auth / client_token
              （缺失属性按未提供处理）
    """
    if args.zmq_endpoint:
        http_config = {}
        provider_config = {}
    else:
        try:
            provider_config = util.load_config(args.config)
            http_config = provider_config['http']
        except FileNotFoundError:
            logger.error("Config file not found: %s", args.config)
            sys.exit(1)
        except KeyError:
            logger.error("Config file doesn't have [http] section: %s", args.config)
            sys.exit(1)
    overrides = {
        'host': getattr(args, 'host', None),
        'port': getattr(args, 'port', None),
        'zmq_endpoint': getattr(args, 'zmq_endpoint', None),
        'zmq_auth_token': getattr(args, 'auth', None),
        'client_tokens': getattr(args, 'client_token', None),
    }
    try:
        config = build_http_config(http_config, overrides)
        asyncio.run(serve(config, provider_config))
    except KeyboardInterrupt:
        logger.info("HTTP server stopped")
    except Exception:
        logger.exception("Can't start HTTP server")
        sys.exit(1)


__all__ = [
    "ErrorResponse", "ErrorEnvelope", "Usage", "ResponseMessage", "Choice",
    "ChatCompletion", "Delta", "ChunkChoice", "ChatCompletionChunk",
    "HttpServerConfig", "build_http_config", "create_app", "serve",
    "run_http", "make_json_response", "make_error_response",
    "parse_openai_messages", "parse_openai_tools",
    "build_chat_completion_response", "build_chat_completion_chunk",
    "build_usage_chunk", "error_response_for_finish_reason",
    "error_envelope_for_finish_reason", "sse_frame", "SSE_DONE",
    "handle_chat_completions", "handle_list_models",
]
