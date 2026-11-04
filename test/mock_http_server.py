"""测试共享辅助：启动真实 aitoolman HTTP 服务端并注入 mock 场景钩子。

启动一个真实的 aitoolman HTTP 服务端实例，经 mock_handler 请求拦截钩子
提供 mock 场景（mock-normal、mock-stream、mock-slow-batch、
mock-slow-stream、mock-heartbeat、mock-heartbeat-content、mock-auth-error、
mock-loop），测试经与生产一致的服务路径访问，无需任何真实上游模型配置。

用法：
    base_url, stop = start_mock_http_server(CLIENT_TOKEN)
    try:
        ...  # 请求 base_url + "/v1/chat/completions"
    finally:
        stop()
"""

import asyncio
import socket
import threading
import time
import urllib.request
from typing import Any, Callable, Dict, List, Optional, Tuple

try:
    from aiohttp import web
except ImportError:
    web = None  # 缺失时由 build_http_config/serve 统一报错

from aitoolman import util
from aitoolman.http_server import (
    SSE_DONE,
    ChatCompletion,
    Choice,
    Delta,
    ResponseMessage,
    Usage,
    build_chat_completion_chunk,
    build_http_config,
    build_usage_chunk,
    make_error_response,
    make_json_response,
    serve,
    sse_frame,
)

# 就绪等待与停止等待的超时时间（秒）
READY_TIMEOUT = 10.0
STOP_TIMEOUT = 10.0


# ---------------------------------------------------------------------------
# mock 场景（测试侧经 mock_handler 钩子注入服务端启用）
# ---------------------------------------------------------------------------

# 场景模型名，按请求的 model 字段匹配（详见 _handle_mock_request）
MOCK_SCENARIOS = frozenset({
    "mock-normal", "mock-stream", "mock-slow-batch", "mock-slow-stream",
    "mock-heartbeat", "mock-heartbeat-content", "mock-auth-error",
    "mock-loop",
})
MOCK_SLOW_BATCH_DELAY = 5.0       # mock-slow-batch 默认延迟（秒）
MOCK_SLOW_STREAM_DELAY = 2.0      # mock-slow-stream 默认暂停（秒）
MOCK_HEARTBEAT_DURATION = 5.0     # mock-heartbeat 纯心跳持续时长（秒）
MOCK_STREAM_CHUNK_INTERVAL = 0.1  # 流式内容块间隔（秒）
MOCK_HEARTBEAT_INTERVAL = 0.1     # 心跳事件间隔（秒）
MOCK_STREAM_CONTENT = ["Hello", " from ", "mock", " server!"]
MOCK_SLOW_STREAM_CONTENT = ["Hello", " from ", "slow", " stream!"]
MOCK_LOOP_TAIL_LENGTH = 20       # mock-loop 循环尾部片段默认字符数
MOCK_LOOP_REPEATS = 64           # mock-loop 循环轮数默认值
MOCK_LOOP_CHUNK_SIZE = 16        # mock-loop 每帧最大字符数默认值
MOCK_LOOP_FALLBACK_TEXT = "Hello from mock-loop!"
# SSE 心跳帧（注释行，标准 SSE 解析器与客户端会自动忽略）
_SSE_HEARTBEAT = b": keep-alive\n\n"


def _get_mock_delay(request: "web.Request", default: float) -> float:
    """读取 X-Mock-Response-Delay 请求头覆盖慢场景延迟（秒），非法值回退默认"""
    header = request.headers.get("X-Mock-Response-Delay")
    if header is None:
        return default
    try:
        return max(0.0, float(header))
    except ValueError:
        return default


def _get_int_header(request: "web.Request", name: str,
                    default: int, minimum: int) -> int:
    """读取整型请求头，缺失、非法（含低于下限）时回退默认值"""
    header = request.headers.get(name)
    if header is None:
        return default
    try:
        value = int(header)
    except ValueError:
        return default
    return value if value >= minimum else default


def _mock_loop_text(raw_messages: Any) -> str:
    """从请求 messages 提取 mock-loop 响应文本。

    content 为字符串的消息按序拼接；数组形式仅取 text part；无有效文本
    时回退固定短文本。
    """
    texts: List[str] = []
    if isinstance(raw_messages, list):
        for raw in raw_messages:
            if not isinstance(raw, dict):
                continue
            content = raw.get("content")
            if isinstance(content, str):
                texts.append(content)
            elif isinstance(content, list):
                for part in content:
                    if isinstance(part, dict) and part.get("type") == "text":
                        texts.append(str(part.get("text") or ""))
    return "".join(texts) or MOCK_LOOP_FALLBACK_TEXT


def _mock_batch_response(selector: str, content: str) -> ChatCompletion:
    """构造 mock 批量响应，model 字段回显请求中的 selector"""
    return ChatCompletion(
        id="chatcmpl-mock-" + util.get_id(),
        object="chat.completion",
        created=int(time.time()),
        model=selector,
        choices=[Choice(
            index=0,
            message=ResponseMessage(role="assistant", content=content),
            finish_reason="stop",
        )],
        usage=Usage(prompt_tokens=10, completion_tokens=20, total_tokens=30),
    )


async def _stream_mock_response(
        request: "web.Request", selector: str,
        stream_options: Optional[Dict[str, Any]],
        loop_text: str = "") -> "web.StreamResponse":
    """mock 流式场景：按 selector 输出 SSE 帧，model 字段回显 selector。

    - mock-stream：正常分片流
    - mock-slow-stream：第二个内容块前暂停（X-Mock-Response-Delay 覆盖，默认 2s）
    - mock-heartbeat：仅心跳事件，持续 MOCK_HEARTBEAT_DURATION 秒后结束
    - mock-heartbeat-content：心跳与内容块交织的正常流
    - mock-loop：先按 X-Mock-Loop-Chunk-Size 逐帧输出 loop_text，再以其尾部
      X-Mock-Loop-Tail-Length 字符为循环串，按同样分帧方式循环输出
      X-Mock-Loop-Repeats 轮（三个请求头均非法值回退默认）
    收尾与真实流式路径一致：finish chunk →（include_usage 时）usage chunk
    → [DONE]；mock-heartbeat 除外（无内容，直接结束）。
    """
    chunk_id = "chatcmpl-mock-" + util.get_id()
    created = int(time.time())
    resp = web.StreamResponse(headers={
        "Content-Type": "text/event-stream",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    })
    await resp.prepare(request)

    def frame(delta: Optional[Delta] = None,
              finish_reason: Optional[str] = None) -> bytes:
        return sse_frame(build_chat_completion_chunk(
            chunk_id, created, selector, delta=delta,
            finish_reason=finish_reason))

    if selector == "mock-heartbeat":
        deadline = time.monotonic() + MOCK_HEARTBEAT_DURATION
        while time.monotonic() < deadline:
            await resp.write(_SSE_HEARTBEAT)
            await asyncio.sleep(MOCK_HEARTBEAT_INTERVAL)
        await resp.write_eof()
        return resp

    await resp.write(frame(Delta(role="assistant")))
    if selector == "mock-loop":
        tail_length = _get_int_header(
            request, "X-Mock-Loop-Tail-Length", MOCK_LOOP_TAIL_LENGTH, 0)
        repeats = _get_int_header(
            request, "X-Mock-Loop-Repeats", MOCK_LOOP_REPEATS, 0)
        chunk_size = _get_int_header(
            request, "X-Mock-Loop-Chunk-Size", MOCK_LOOP_CHUNK_SIZE, 1)
        # 先逐帧输出传入文本，再循环输出其尾部片段
        for i in range(0, len(loop_text), chunk_size):
            await resp.write(
                frame(Delta(content=loop_text[i:i + chunk_size])))
        tail = loop_text[-tail_length:] if tail_length else ""
        for _ in range(repeats):
            for i in range(0, len(tail), chunk_size):
                await resp.write(
                    frame(Delta(content=tail[i:i + chunk_size])))
    elif selector == "mock-heartbeat-content":
        for part in MOCK_STREAM_CONTENT:
            await resp.write(_SSE_HEARTBEAT)
            await asyncio.sleep(MOCK_HEARTBEAT_INTERVAL)
            await resp.write(frame(Delta(content=part)))
            await asyncio.sleep(MOCK_STREAM_CHUNK_INTERVAL)
    else:
        parts = (MOCK_SLOW_STREAM_CONTENT
                 if selector == "mock-slow-stream" else MOCK_STREAM_CONTENT)
        pause = (_get_mock_delay(request, MOCK_SLOW_STREAM_DELAY)
                 if selector == "mock-slow-stream" else 0.0)
        for i, part in enumerate(parts):
            if i == 1 and pause:
                await asyncio.sleep(pause)
            await resp.write(frame(Delta(content=part)))
            await asyncio.sleep(MOCK_STREAM_CHUNK_INTERVAL)
    await resp.write(frame(finish_reason="stop"))
    if (stream_options or {}).get("include_usage"):
        await resp.write(sse_frame(build_usage_chunk(
            chunk_id, created, selector, 10, 20)))
    await resp.write(SSE_DONE)
    await resp.write_eof()
    return resp


async def _handle_mock_request(
        request: "web.Request", body: Dict[str, Any],
        selector: str) -> "web.StreamResponse":
    """按请求 model 名分派 mock 场景（认证之后、resolve_model 之前拦截）。

    - mock-auth-error：返回 401，模拟上游认证失败
    - mock-slow-batch：延迟后返回批量响应（X-Mock-Response-Delay 覆盖，默认 5s）
    - mock-normal 与 stream=false 的 mock-stream：批量响应
    - mock-loop：文本取自请求 messages；stream=false 返回传入文本与尾部
      循环内容一次性拼接的批量响应，stream=true 见 _stream_mock_response
    - 其余 stream=true 请求：见 _stream_mock_response
    """
    if selector == "mock-auth-error":
        return make_error_response(
            "Invalid authentication credentials",
            "authentication_error", code="invalid_api_key", status=401)
    if selector == "mock-slow-batch":
        await asyncio.sleep(_get_mock_delay(request, MOCK_SLOW_BATCH_DELAY))
        return make_json_response(
            _mock_batch_response(selector, "This is a slow batch response."))
    if selector == "mock-loop":
        text = _mock_loop_text(body.get("messages"))
        if not body.get("stream"):
            tail_length = _get_int_header(
                request, "X-Mock-Loop-Tail-Length", MOCK_LOOP_TAIL_LENGTH, 0)
            repeats = _get_int_header(
                request, "X-Mock-Loop-Repeats", MOCK_LOOP_REPEATS, 0)
            tail = text[-tail_length:] if tail_length else ""
            return make_json_response(
                _mock_batch_response(selector, text + tail * repeats))
        stream_options = body.get("stream_options")
        return await _stream_mock_response(
            request, selector,
            stream_options if isinstance(stream_options, dict) else None,
            loop_text=text)
    if not body.get("stream") or selector == "mock-normal":
        return make_json_response(
            _mock_batch_response(selector, "This is a mock response."))
    stream_options = body.get("stream_options")
    return await _stream_mock_response(
        request, selector,
        stream_options if isinstance(stream_options, dict) else None)


async def mock_handler(
        request: "web.Request",
        body: Dict[str, Any]) -> Optional["web.StreamResponse"]:
    """服务端 chat/completions 请求拦截钩子（认证之后、resolve_model 之前调用）。

    请求 model 命中 MOCK_SCENARIOS 时委托 _handle_mock_request 返回 mock
    响应；未命中返回 None，放行进入正常业务路径。
    """
    selector = body.get("model")
    if selector in MOCK_SCENARIOS:
        return await _handle_mock_request(request, body, selector)
    return None


def _get_free_port() -> int:
    """返回 127.0.0.1 上一个空闲 TCP 端口。"""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _wait_for_ready(base_url: str, thread: threading.Thread,
                    timeout: float = READY_TIMEOUT) -> None:
    """轮询健康检查端点直至服务就绪；线程提前退出或超时则报错。"""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if not thread.is_alive():
            raise RuntimeError("mock HTTP server exited during startup")
        try:
            with urllib.request.urlopen(base_url + "/health",
                                        timeout=0.5) as resp:
                if resp.status == 200:
                    return
        except Exception:
            time.sleep(0.05)
    raise RuntimeError(
        f"mock HTTP server did not become ready within {timeout}s")


def start_mock_http_server(
        client_token: str,
        host: str = "127.0.0.1") -> Tuple[str, Callable[[], None]]:
    """在随机空闲端口启动注入 mock 场景钩子的 HTTP 服务端并等待就绪。

    Args:
        client_token: 客户端访问所需的 Bearer 令牌；mock 请求经 mock_handler
            钩子在 resolve_model 之前拦截，无需真实模型配置。
        host: 服务绑定地址。

    Returns:
        (base_url, stop)：base_url 形如 "http://127.0.0.1:PORT"；
        stop() 置位停止事件并等待服务线程优雅退出，可重复调用。
    """
    port = _get_free_port()
    base_url = f"http://{host}:{port}"
    config = build_http_config({
        "host": host,
        "port": port,
        "client_tokens": [client_token],
    })
    provider_config = {"server": {}, "default": {}, "api": {}}
    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(
            serve(config, provider_config, stop_event,
                  mock_handler=mock_handler)),
        name="mock-http-server",
        daemon=True,
    )
    thread.start()
    _wait_for_ready(base_url, thread)

    def stop() -> None:
        stop_event.set()
        thread.join(timeout=STOP_TIMEOUT)
        if thread.is_alive():
            raise RuntimeError("mock HTTP server did not stop in time")

    return base_url, stop
