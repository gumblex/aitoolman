"""HTTP Server 骨架测试。

覆盖 aitoolman/http_server.py 的骨架功能：
- build_http_config：默认值、[http] 节读取、CLI 覆盖优先级、client_tokens 合并去重、fail-fast 校验
- Bearer 认证中间件：白名单 / 与 /health 放行、缺失/无效 token 返回 401、有效 token 放行
- GET /health：返回 {"status":"ok","mode":"direct"|"remote"}
- serve 直连模式生命周期：后台线程 LLMZmqServer 启动、/health 可访问、stop_event 置位后优雅停止
"""

import asyncio
import json
import socket
import threading
import time
import unittest
import uuid
from unittest.mock import patch

try:
    import aiohttp
    from aiohttp.test_utils import TestClient, TestServer
    AIOHTTP_AVAILABLE = True
except ImportError:
    AIOHTTP_AVAILABLE = False

from test.mock_http_server import mock_handler, start_mock_http_server

from aitoolman.channel import ChannelEvent
from aitoolman.http_server import (
    HttpServerConfig,
    build_http_config,
    create_app,
    error_response_for_finish_reason,
    parse_openai_messages,
    parse_openai_tools,
    serve,
)
from aitoolman.model import (
    LLMContentFilterError,
    LLMLengthLimitError,
    LLMNoAvailableModelError,
    LLMProviderResponse,
    ModelInfo,
    ToolCall,
)
from aitoolman.zmqserver import LLMZmqServer

CLIENT_TOKEN = "test-secret-token"


def _get_free_port() -> int:
    """Return a free TCP port on 127.0.0.1."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return sock.getsockname()[1]


def _make_http_config(**overrides) -> HttpServerConfig:
    """构造测试用 HttpServerConfig（默认直连模式、固定 client token）"""
    params = dict(
        host="127.0.0.1",
        port=0,
        zmq_endpoint=None,
        zmq_auth_token=None,
        client_tokens=[CLIENT_TOKEN],
    )
    params.update(overrides)
    return HttpServerConfig(**params)


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestBuildHttpConfig(unittest.TestCase):
    """build_http_config：默认值、配置节读取、CLI 覆盖与 fail-fast 校验"""

    def test_defaults(self):
        # client_tokens 为必填项（为空时 fail-fast，由 test_empty_client_tokens_raises 覆盖），
        # 此处提供合法 token 以验证其余字段的内置默认值
        config = build_http_config({"client_tokens": [CLIENT_TOKEN]})
        self.assertEqual(config.host, "127.0.0.1")
        self.assertEqual(config.port, 8000)
        self.assertIsNone(config.zmq_endpoint)
        self.assertIsNone(config.zmq_auth_token)
        self.assertEqual(config.client_tokens, [CLIENT_TOKEN])
        self.assertEqual(config.mode, "direct")

    def test_http_section(self):
        config = build_http_config({
            "host": "0.0.0.0",
            "port": 9000,
            "client_tokens": ["tok-a"],
        })
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)
        self.assertEqual(config.client_tokens, ["tok-a"])
        self.assertEqual(config.mode, "direct")

    def test_remote_mode(self):
        config = build_http_config({
            "zmq_endpoint": "tcp://127.0.0.1:5555",
            "zmq_auth_token": "internal-secret",
            "client_tokens": ["tok-a"],
        })
        self.assertEqual(config.zmq_endpoint, "tcp://127.0.0.1:5555")
        self.assertEqual(config.zmq_auth_token, "internal-secret")
        self.assertEqual(config.mode, "remote")

    def test_cli_overrides_section(self):
        config = build_http_config(
            {"host": "0.0.0.0", "port": 9000, "client_tokens": ["tok-a"]},
            overrides={
                "host": "192.168.1.1",
                "port": 1234,
                "zmq_endpoint": "tcp://10.0.0.1:6000",
                "zmq_auth_token": "cli-token",
            },
        )
        self.assertEqual(config.host, "192.168.1.1")
        self.assertEqual(config.port, 1234)
        self.assertEqual(config.zmq_endpoint, "tcp://10.0.0.1:6000")
        self.assertEqual(config.zmq_auth_token, "cli-token")
        self.assertEqual(config.mode, "remote")

    def test_client_tokens_merge_dedup(self):
        config = build_http_config(
            {"client_tokens": ["tok-a", "tok-b"]},
            overrides={"client_tokens": ["tok-b", "tok-c", ""]},
        )
        self.assertEqual(config.client_tokens, ["tok-a", "tok-b", "tok-c"])

    def test_none_overrides_keep_section_values(self):
        config = build_http_config(
            {"host": "0.0.0.0", "port": 9000, "client_tokens": ["tok-a"]},
            overrides={"host": None, "port": None},
        )
        self.assertEqual(config.host, "0.0.0.0")
        self.assertEqual(config.port, 9000)

    def test_empty_client_tokens_raises(self):
        with self.assertRaises(ValueError):
            build_http_config({})
        with self.assertRaises(ValueError):
            build_http_config({"client_tokens": ["", ""]})

    def test_remote_missing_auth_token_raises(self):
        with self.assertRaises(ValueError):
            build_http_config({
                "zmq_endpoint": "tcp://127.0.0.1:5555",
                "client_tokens": ["tok-a"],
            })


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestAuthMiddleware(unittest.IsolatedAsyncioTestCase):
    """Bearer 认证中间件与 /health 端点（TestClient 进程内验证）"""

    async def _request(self, config, path, headers=None):
        """创建 app 并发起 GET 请求，返回 (status, headers, json 或 None)"""
        app = create_app(config, client=None)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(path, headers=headers or {})
            body = await resp.read()
            try:
                data = json.loads(body)
            except ValueError:
                data = None
            return resp.status, resp.headers, data

    async def test_health_direct_mode_no_auth(self):
        status, _, data = await self._request(_make_http_config(), "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data, {"status": "ok", "mode": "direct"})

    async def test_health_remote_mode_no_auth(self):
        config = _make_http_config(
            zmq_endpoint="tcp://127.0.0.1:5555", zmq_auth_token="internal")
        status, _, data = await self._request(config, "/health")
        self.assertEqual(status, 200)
        self.assertEqual(data, {"status": "ok", "mode": "remote"})

    async def test_root_whitelist_no_auth(self):
        status, _, data = await self._request(_make_http_config(), "/")
        self.assertEqual(status, 200)
        self.assertEqual(data["status"], "ok")

    async def test_missing_token_returns_401(self):
        status, headers, data = await self._request(
            _make_http_config(), "/v1/models")
        self.assertEqual(status, 401)
        self.assertEqual(headers.get("WWW-Authenticate"), "Bearer")
        self.assertEqual(data["error"]["type"], "authentication_error")
        self.assertEqual(data["error"]["code"], "invalid_api_key")
        self.assertTrue(data["error"]["message"])

    async def test_invalid_token_returns_401(self):
        status, _, _ = await self._request(
            _make_http_config(), "/v1/chat/completions",
            headers={"Authorization": "Bearer wrong-token"})
        self.assertEqual(status, 401)

    async def test_valid_token_passes_auth(self):
        # 认证通过后应到达路由层（未注册路径返回 404 而非 401）
        status, _, _ = await self._request(
            _make_http_config(), "/v1/nonexistent",
            headers={"Authorization": f"Bearer {CLIENT_TOKEN}"})
        self.assertEqual(status, 404)


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestServeDirectLifecycle(unittest.IsolatedAsyncioTestCase):
    """直连模式 serve 生命周期：后台线程 ZMQ server + /health 可访问 + 优雅停止"""

    async def _wait_for_health(self, base_url, timeout=10.0):
        """轮询 /health 直到返回 200，返回响应 JSON"""
        deadline = time.monotonic() + timeout
        last_error = None
        while time.monotonic() < deadline:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                            base_url + "/health",
                            timeout=aiohttp.ClientTimeout(total=1)) as resp:
                        if resp.status == 200:
                            return await resp.json()
            except Exception as e:
                last_error = e
            await asyncio.sleep(0.1)
        self.fail(f"/health not ready within {timeout}s: last_error={last_error!r}")

    async def test_serve_direct_start_and_stop(self):
        port = _get_free_port()
        config = build_http_config(
            {"port": port, "client_tokens": [CLIENT_TOKEN]})
        provider_config = {"server": {}, "default": {}, "api": {}}
        stop_event = threading.Event()
        task = asyncio.create_task(serve(config, provider_config, stop_event))
        try:
            data = await self._wait_for_health(f"http://127.0.0.1:{port}")
            self.assertEqual(data["status"], "ok")
            self.assertEqual(data["mode"], "direct")
        finally:
            stop_event.set()
            # serve 应在轮询间隔与清理流程内返回，不挂起
            await asyncio.wait_for(task, timeout=15.0)


# ---------------------------------------------------------------------------
# OpenAI 协议转换纯函数
# ---------------------------------------------------------------------------

class TestParseOpenAIMessages(unittest.TestCase):
    """OpenAI messages 数组 → Message 转换"""

    def test_string_content(self):
        result = parse_openai_messages([{"role": "user", "content": "hi"}])
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0].role, "user")
        self.assertEqual(result[0].content, "hi")
        self.assertIsNone(result[0].media_content)
        self.assertIsNone(result[0].raw_value)

    def test_text_parts_joined(self):
        result = parse_openai_messages([{
            "role": "user",
            "content": [{"type": "text", "text": "a"},
                        {"type": "text", "text": "b"}],
        }])
        self.assertEqual(result[0].content, "ab")

    def test_image_part(self):
        result = parse_openai_messages([{
            "role": "user",
            "content": [
                {"type": "text", "text": "look"},
                {"type": "image_url",
                 "image_url": {"url": "https://example.com/x.png"}},
            ],
        }])
        self.assertEqual(result[0].content, "look")
        self.assertEqual(len(result[0].media_content), 1)
        self.assertEqual(result[0].media_content[0].media_type, "image")
        self.assertEqual(result[0].media_content[0].url,
                         "https://example.com/x.png")

    def test_unknown_part_raw_passthrough(self):
        raw = {"role": "user",
               "content": [{"type": "audio", "audio": {"data": "..."}}]}
        result = parse_openai_messages([raw])
        self.assertEqual(result[0].raw_value, raw)

    def test_tool_message(self):
        result = parse_openai_messages([
            {"role": "tool", "content": "42", "tool_call_id": "tc-1"}])
        self.assertEqual(result[0].role, "tool")
        self.assertEqual(result[0].content, "42")
        self.assertEqual(result[0].tool_call_id, "tc-1")

    def test_assistant_tool_calls_passthrough(self):
        raw = {"role": "assistant", "content": None, "tool_calls": [
            {"id": "call-1", "type": "function",
             "function": {"name": "f", "arguments": "{}"}}]}
        result = parse_openai_messages([raw])
        self.assertEqual(result[0].raw_value, raw)

    def test_invalid_message_raises(self):
        with self.assertRaises(ValueError):
            parse_openai_messages(["not-a-dict"])
        with self.assertRaises(ValueError):
            parse_openai_messages([{"content": "missing role"}])


class TestParseOpenAITools(unittest.TestCase):
    """OpenAI tools 数组 → aitoolman 工具配置转换"""

    def test_convert(self):
        result = parse_openai_tools([{
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "query weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "city": {"type": "string",
                                 "description": "city name"},
                        "unit": {"type": "string"},
                    },
                    "required": ["city"],
                },
            },
        }])
        self.assertEqual(list(result), ["get_weather"])
        tool = result["get_weather"]
        self.assertEqual(tool["type"], "function")
        self.assertEqual(tool["description"], "query weather")
        self.assertEqual(tool["param"]["city"],
                         {"type": "string", "description": "city name",
                          "required": True})
        self.assertEqual(tool["param"]["unit"],
                         {"type": "string", "description": ""})

    def test_missing_function_raises(self):
        with self.assertRaises(ValueError):
            parse_openai_tools([{"type": "function"}])

    def test_missing_name_raises(self):
        with self.assertRaises(ValueError):
            parse_openai_tools([{"type": "function",
                                 "function": {"description": "x"}}])

    def test_invalid_parameters_raises(self):
        # parameters 非 dict
        with self.assertRaises(ValueError):
            parse_openai_tools([{
                "type": "function",
                "function": {"name": "f", "parameters": "not-a-dict"},
            }])
        # properties 非 dict
        with self.assertRaises(ValueError):
            parse_openai_tools([{
                "type": "function",
                "function": {"name": "f", "parameters": {
                    "properties": "not-a-dict"}},
            }])
        # required 非 list
        with self.assertRaises(ValueError):
            parse_openai_tools([{
                "type": "function",
                "function": {"name": "f", "parameters": {
                    "properties": {}, "required": "city"}},
            }])


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestFinishReasonErrorMapping(unittest.TestCase):
    """finish_reason 错误路径映射"""

    def test_ok_reasons_return_none(self):
        for reason in ("stop", "length", "tool_calls", "content_filter", None):
            self.assertIsNone(error_response_for_finish_reason(reason, None))

    def test_request_error_maps_502(self):
        resp = error_response_for_finish_reason("error: request",
                                                "upstream boom")
        self.assertEqual(resp.status, 502)

    def test_other_error_maps_500(self):
        resp = error_response_for_finish_reason("error: format", "bad parse")
        self.assertEqual(resp.status, 500)


# ---------------------------------------------------------------------------
# 阻塞 chat/completions 与 /v1/models（fake client 注入）
# ---------------------------------------------------------------------------

class _FakeProviderRequest:
    """携带已完成 response Future 的最小请求替身"""

    def __init__(self, response_future):
        self.response = response_future


class _FakeStreamRequest:
    """流式请求替身：携带 request_id / response Future / 输出通道"""

    def __init__(self, request_id, response_future, output_channel):
        self.request_id = request_id
        self.response = response_future
        self.output_channel = output_channel


class _FakeZmqClient:
    """LLMZmqClient 替身：记录请求参数并返回预置响应。

    stream_events 非 None 时，stream=True 的请求由后台任务把事件依次写入
    output_channel 后置最终响应；默认经 write_complete 收尾（EOFError 路径），
    end_with_none_event 为真时模拟真实 ZMQ 链路：不传播 EOF，改以
    (reasoning, None) 与 (response, None) 结束事件收尾；pause_after 与
    stream_pause 配合可在写完指定数量的事件后暂停泵（断连场景使用）。
    """

    def __init__(self, response=None, candidates=None, resolve_error=None,
                 models=None, tags=None, stream_events=None, pause_after=None,
                 stream_pause=None, end_with_none_event=False):
        self.response = response
        self.candidates = candidates if candidates is not None else ["model-a"]
        self.resolve_error = resolve_error
        self.models = models if models is not None else []
        self.tags = tags if tags is not None else {}
        self.stream_events = stream_events
        self.pause_after = pause_after
        self.stream_pause = stream_pause
        self.end_with_none_event = end_with_none_event
        self.resolve_calls = []
        self.requests = []
        self.cancel_calls = []
        self.stream_tasks = []

    async def resolve_model(self, tags, messages=None, context_id=None):
        self.resolve_calls.append({"tags": tags, "messages": messages})
        if self.resolve_error is not None:
            raise self.resolve_error
        return list(self.candidates)

    async def request(self, model_name, messages, tools=None, options=None,
                      stream=False, context_id=None, output_channel=None):
        self.requests.append({
            "model_name": model_name,
            "messages": messages,
            "tools": tools,
            "options": options,
            "stream": stream,
            "output_channel": output_channel,
        })
        future = asyncio.get_running_loop().create_future()
        if stream and self.stream_events is not None:
            stream_request = _FakeStreamRequest(
                "req-stream-0001", future, output_channel)
            task = asyncio.get_running_loop().create_task(
                self._pump(stream_request))
            self.stream_tasks.append(task)
            return stream_request
        future.set_result(self.response)
        return _FakeProviderRequest(future)

    async def _pump(self, stream_request):
        """按预置事件序列驱动输出通道，结束后写入最终响应"""
        count = 0
        for event in self.stream_events:
            if (self.stream_pause is not None
                    and self.pause_after is not None
                    and count == self.pause_after):
                await self.stream_pause.wait()
            await stream_request.output_channel.write(event)
            count += 1
        if self.end_with_none_event:
            # 真实 ZMQ 链路不传播 EOF：provider 收尾写入 (reasoning, None)
            # 与 (response, None)，流式泵应跳过前者、遇后者退出循环
            await stream_request.output_channel.write(
                ChannelEvent("reasoning", None))
            await stream_request.output_channel.write(
                ChannelEvent("response", None))
        else:
            await stream_request.output_channel.write_complete()
        if not stream_request.response.done():
            stream_request.response.set_result(self.response)

    async def cancel(self, request_id):
        self.cancel_calls.append(request_id)

    async def list_models(self, tag=None):
        return list(self.models)

    async def list_tags(self):
        return dict(self.tags)


def _make_provider_response(**overrides):
    params = dict(
        client_id="test-client",
        context_id=None,
        request_id="req-0001",
        model_name="model-a",
        stream=False,
        response_text="Hello!",
        finish_reason="stop",
        prompt_tokens=10,
        completion_tokens=5,
    )
    params.update(overrides)
    return LLMProviderResponse(**params)


def _make_model_info(name="model-a"):
    return ModelInfo(
        name=name, url="http://127.0.0.1:9/v1", model="gpt-test",
        parallel=1, timeout=30.0, timeout_batch=60.0, api_type="openai",
        body_options={}, max_input_tokens=None, bytes_per_token=4.0,
        tags={"chat": 1.0},
    )


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestChatCompletionsBlocking(unittest.IsolatedAsyncioTestCase):
    """POST /v1/chat/completions 阻塞路径（fake client 注入）"""

    async def _post_chat(self, fake_client, body):
        app = create_app(_make_http_config(), client=fake_client)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                data=body if isinstance(body, str) else json.dumps(body),
                headers={"Authorization": "Bearer " + CLIENT_TOKEN})
            return resp.status, resp.headers, json.loads(await resp.read())

    async def test_success_batch(self):
        fake = _FakeZmqClient(response=_make_provider_response())
        status, headers, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.7,
            "max_tokens": 128,
        })
        self.assertEqual(status, 200)
        self.assertTrue(data["id"].startswith("chatcmpl-"))
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "model-a")
        choice = data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"], "Hello!")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(data["usage"], {"prompt_tokens": 10,
                                         "completion_tokens": 5,
                                         "total_tokens": 15})
        self.assertEqual(headers.get("x-aitoolman-resolved-model"), "model-a")
        # model 原样传入 resolve_model，候选首个作为实际模型
        self.assertEqual(fake.resolve_calls[0]["tags"], "chat")
        req = fake.requests[0]
        self.assertEqual(req["model_name"], "model-a")
        self.assertIs(req["stream"], False)
        self.assertEqual(req["tools"], {})
        # 保留键不进 options，其余全量透传
        self.assertEqual(req["options"],
                         {"temperature": 0.7, "max_tokens": 128})
        self.assertEqual(req["messages"][0].role, "user")
        self.assertEqual(req["messages"][0].content, "hi")

    async def test_reasoning_content_included(self):
        fake = _FakeZmqClient(response=_make_provider_response(
            response_text="answer", response_reasoning="thinking..."))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 200)
        self.assertEqual(data["choices"][0]["message"]["reasoning_content"],
                         "thinking...")

    async def test_missing_model_returns_400(self):
        fake = _FakeZmqClient()
        status, _, data = await self._post_chat(
            fake, {"messages": [{"role": "user", "content": "hi"}]})
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "invalid_request_error")
        self.assertEqual(fake.requests, [])

    async def test_invalid_json_returns_400(self):
        fake = _FakeZmqClient()
        status, _, data = await self._post_chat(fake, "not-json")
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "invalid_request_error")

    async def test_invalid_messages_returns_400(self):
        fake = _FakeZmqClient()
        status, _, data = await self._post_chat(
            fake, {"model": "chat", "messages": "hi"})
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "invalid_request_error")

    async def test_invalid_tools_returns_400(self):
        fake = _FakeZmqClient()
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function"}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "invalid_request_error")

    async def test_model_not_found_returns_404(self):
        fake = _FakeZmqClient(
            resolve_error=LLMNoAvailableModelError("no model for tags"))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 404)
        self.assertEqual(data["error"]["type"], "model_not_found")

    async def test_length_limit_error_returns_400(self):
        fake = _FakeZmqClient(
            resolve_error=LLMLengthLimitError("context too long"))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "context_length_exceeded")

    async def test_content_filter_error_returns_400(self):
        fake = _FakeZmqClient(
            resolve_error=LLMContentFilterError("blocked by policy"))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 400)
        self.assertEqual(data["error"]["type"], "content_filter")

    async def test_upstream_error_returns_502(self):
        fake = _FakeZmqClient(response=_make_provider_response(
            finish_reason="error: request", error_text="upstream boom"))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 502)
        self.assertEqual(data["error"]["type"], "upstream_error")
        self.assertEqual(data["error"]["message"], "upstream boom")

    async def test_tool_calls_response(self):
        fake = _FakeZmqClient(response=_make_provider_response(
            response_text="",
            finish_reason="tool_calls",
            response_tool_calls=[
                ToolCall(name="get_weather",
                         arguments_text='{"city": "SF"}',
                         arguments={"city": "SF"}, id="call-1")],
        ))
        status, _, data = await self._post_chat(fake, {
            "model": "chat",
            "messages": [{"role": "user", "content": "weather?"}],
        })
        self.assertEqual(status, 200)
        message = data["choices"][0]["message"]
        self.assertEqual(data["choices"][0]["finish_reason"], "tool_calls")
        self.assertEqual(message["tool_calls"][0]["id"], "call-1")
        self.assertEqual(message["tool_calls"][0]["type"], "function")
        self.assertEqual(message["tool_calls"][0]["function"]["name"],
                         "get_weather")
        self.assertEqual(message["tool_calls"][0]["function"]["arguments"],
                         '{"city": "SF"}')

    async def test_list_models(self):
        fake = _FakeZmqClient(
            models=[_make_model_info("model-a"),
                    _make_model_info("model-b")],
            tags={
                "chat": [_make_model_info("model-a")],
                "model-b": [_make_model_info("model-b")],
                "empty": [],
            })
        app = create_app(_make_http_config(), client=fake)
        async with TestClient(TestServer(app)) as client:
            resp = await client.get(
                "/v1/models",
                headers={"Authorization": "Bearer " + CLIENT_TOKEN})
            self.assertEqual(resp.status, 200)
            data = await resp.json()
        self.assertEqual(data["object"], "list")
        # 模型条目在前，标签条目追加在后；与模型重名的标签跳过，空标签不输出
        self.assertEqual([m["id"] for m in data["data"]],
                         ["model-a", "model-b", "chat"])
        self.assertEqual(data["data"][0]["object"], "model")
        self.assertEqual(data["data"][0]["owned_by"], "aitoolman")


# ---------------------------------------------------------------------------
# stream=true SSE 流式路径（fake client 注入 + 通道泵）
# ---------------------------------------------------------------------------

def _parse_sse_frames(body: bytes) -> list:
    """把 SSE 响应体解析为帧列表：JSON 帧返回 dict，终止帧返回 "[DONE]" """
    frames = []
    for block in body.split(b"\n\n"):
        if not block:
            continue
        if block == b"data: [DONE]":
            frames.append("[DONE]")
            continue
        assert block.startswith(b"data: "), repr(block)
        frames.append(json.loads(block[len(b"data: "):]))
    return frames


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestChatCompletionsStreaming(unittest.IsolatedAsyncioTestCase):
    """stream=true：delta 序列、finish/usage chunk、[DONE] 终止与断连取消"""

    def _stream_body(self, **extra):
        body = {
            "model": "chat",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        }
        body.update(extra)
        return body

    async def _post_stream(self, fake_client, body):
        """进程内发起流式请求，返回解析后的帧列表"""
        app = create_app(_make_http_config(), client=fake_client)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                data=json.dumps(body),
                headers={"Authorization": "Bearer " + CLIENT_TOKEN})
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream",
                          resp.headers.get("Content-Type", ""))
            self.assertEqual(resp.headers.get("x-aitoolman-resolved-model"),
                             "model-a")
            raw = await resp.read()
        return _parse_sse_frames(raw)

    async def test_delta_sequence_and_finish(self):
        fake = _FakeZmqClient(
            stream_events=[
                ChannelEvent("response", "Hel"),
                ChannelEvent("response", "lo"),
                ChannelEvent("reasoning", "thinking"),
                ChannelEvent("response", " world"),
            ],
            response=_make_provider_response(
                response_text="Hello world",
                response_reasoning="thinking",
                finish_reason="stop",
            ),
        )
        frames = await self._post_stream(fake, self._stream_body())
        self.assertEqual(frames[-1], "[DONE]")
        body_frames = frames[:-1]
        self.assertEqual(len(body_frames), 5)
        for frame in body_frames:
            self.assertEqual(frame["object"], "chat.completion.chunk")
            self.assertTrue(frame["id"].startswith("chatcmpl-"))
            self.assertEqual(frame["model"], "model-a")
        # delta 帧按事件写入顺序透出 content / reasoning_content
        self.assertEqual(
            [(f["choices"][0]["delta"].get("content"),
              f["choices"][0]["delta"].get("reasoning_content"))
             for f in body_frames[:4]],
            [("Hel", None), ("lo", None), (None, "thinking"),
             (" world", None)])
        # 第 5 帧为 finish chunk（空 delta），其后无 usage chunk
        finish_frame = body_frames[-1]
        self.assertEqual(finish_frame["choices"][0]["finish_reason"], "stop")
        self.assertNotIn("usage", finish_frame)

    async def test_usage_chunk_when_include_usage(self):
        fake = _FakeZmqClient(
            stream_events=[ChannelEvent("response", "hi")],
            response=_make_provider_response(finish_reason="stop"),
        )
        frames = await self._post_stream(fake, self._stream_body(
            stream_options={"include_usage": True}))
        self.assertEqual(frames[-1], "[DONE]")
        # [DONE] 前为空 choices 的 usage chunk，再往前为 finish chunk
        usage_frame = frames[-2]
        self.assertEqual(usage_frame["choices"], [])
        self.assertEqual(usage_frame["usage"],
                         {"prompt_tokens": 10, "completion_tokens": 5,
                          "total_tokens": 15})
        self.assertEqual(frames[-3]["choices"][0]["finish_reason"], "stop")

    async def test_no_usage_chunk_by_default(self):
        fake = _FakeZmqClient(
            stream_events=[ChannelEvent("response", "hi")],
            response=_make_provider_response(finish_reason="stop"),
        )
        frames = await self._post_stream(fake, self._stream_body())
        self.assertEqual(frames[-1], "[DONE]")
        self.assertEqual(frames[-2]["choices"][0]["finish_reason"], "stop")
        for frame in frames[:-1]:
            self.assertNotIn("usage", frame)

    async def test_stream_error_envelope_then_done(self):
        fake = _FakeZmqClient(
            stream_events=[ChannelEvent("response", "partial")],
            response=_make_provider_response(
                finish_reason="error: request", error_text="upstream boom"),
        )
        frames = await self._post_stream(fake, self._stream_body())
        self.assertEqual(frames[-1], "[DONE]")
        error_frame = frames[-2]
        self.assertEqual(error_frame["error"]["type"], "upstream_error")
        self.assertEqual(error_frame["error"]["message"], "upstream boom")

    async def test_stream_end_with_none_event(self):
        """模拟真实 ZMQ 链路：不经 write_complete，而以 (response, None)
        结束事件收尾；泵应跳过 reasoning 的 None 结束事件、遇 response 的
        None 结束事件退出并正常发送 finish/[DONE]"""
        fake = _FakeZmqClient(
            stream_events=[ChannelEvent("response", "Hel"),
                           ChannelEvent("reasoning", "hmm"),
                           ChannelEvent("response", "lo")],
            end_with_none_event=True,
            response=_make_provider_response(
                response_text="Hello",
                response_reasoning="hmm",
                finish_reason="stop",
            ),
        )
        frames = await asyncio.wait_for(
            self._post_stream(fake, self._stream_body()), timeout=10)
        self.assertEqual(frames[-1], "[DONE]")
        body_frames = frames[:-1]
        self.assertEqual(
            [(f["choices"][0]["delta"].get("content"),
              f["choices"][0]["delta"].get("reasoning_content"))
             for f in body_frames[:-1]],
            [("Hel", None), (None, "hmm"), ("lo", None)])
        self.assertEqual(body_frames[-1]["choices"][0]["finish_reason"],
                         "stop")

    async def test_client_disconnect_cancels_upstream(self):
        """真实 TCP：读到部分流后断开连接，服务端应取消上游请求"""
        fake = _FakeZmqClient(
            stream_events=[
                ChannelEvent("response", "first"),
                ChannelEvent("response", "second"),
                ChannelEvent("response", "third"),
            ],
            pause_after=2,
            stream_pause=asyncio.Event(),
            response=_make_provider_response(finish_reason="stop"),
        )
        app = create_app(_make_http_config(), client=fake)
        server = TestServer(app)
        await server.start_server()
        try:
            session = aiohttp.ClientSession(
                timeout=aiohttp.ClientTimeout(total=10))
            try:
                resp = await session.post(
                    server.make_url("/v1/chat/completions"),
                    json=self._stream_body(),
                    headers={"Authorization": "Bearer " + CLIENT_TOKEN})
                self.assertEqual(resp.status, 200)
                # 读取包含 "second" 的部分流后突然断开连接
                seen = b""
                while b'"second"' not in seen:
                    line = await resp.content.readline()
                    if not line:
                        break
                    seen += line
                self.assertIn(b'"first"', seen)
                resp.close()
            finally:
                await session.close()
            # 放行事件泵，服务端应感知断连并取消上游请求
            fake.stream_pause.set()
            deadline = time.monotonic() + 5.0
            while not fake.cancel_calls and time.monotonic() < deadline:
                await asyncio.sleep(0.05)
            self.assertEqual(fake.cancel_calls, ["req-stream-0001"])
        finally:
            fake.stream_pause.set()
            if fake.stream_tasks:
                await asyncio.gather(*fake.stream_tasks,
                                     return_exceptions=True)
            await server.close()


# ---------------------------------------------------------------------------
# mock 场景：经 mock_handler 钩子按 model 名拦截
# ---------------------------------------------------------------------------

@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp not installed")
class TestMockScenarios(unittest.IsolatedAsyncioTestCase):
    """mock 场景集成：批量/流式断言、延迟头生效、mock 请求同样经过认证"""

    async def _post(self, body, config=None, headers=None):
        """进程内发起 chat/completions 请求，返回 (status, headers, 原始字节)"""
        app = create_app(config or _make_http_config(), client=None,
                         mock_handler=mock_handler)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                data=json.dumps(body),
                headers=headers if headers is not None
                else {"Authorization": "Bearer " + CLIENT_TOKEN})
            return resp.status, resp.headers, await resp.read()

    async def test_mock_normal_batch(self):
        status, _, raw = await self._post({
            "model": "mock-normal",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertTrue(data["id"].startswith("chatcmpl-"))
        # model 字段回显请求中的 selector
        self.assertEqual(data["model"], "mock-normal")
        self.assertEqual(data["choices"][0]["message"]["content"],
                         "This is a mock response.")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["usage"], {"prompt_tokens": 10,
                                         "completion_tokens": 20,
                                         "total_tokens": 30})

    async def test_mock_disabled_treats_as_normal_model(self):
        # 未注册 mock 钩子时，mock-* 按普通模型名走 resolve_model
        fake = _FakeZmqClient(
            resolve_error=LLMNoAvailableModelError("no model"))
        app = create_app(_make_http_config(), client=fake)
        async with TestClient(TestServer(app)) as client:
            resp = await client.post(
                "/v1/chat/completions",
                data=json.dumps({
                    "model": "mock-normal",
                    "messages": [{"role": "user", "content": "hi"}],
                }),
                headers={"Authorization": "Bearer " + CLIENT_TOKEN})
            self.assertEqual(resp.status, 404)
            data = json.loads(await resp.read())
        self.assertEqual(data["error"]["type"], "model_not_found")

    async def test_mock_auth_error(self):
        status, _, raw = await self._post({
            "model": "mock-auth-error",
            "messages": [{"role": "user", "content": "hi"}],
        })
        self.assertEqual(status, 401)
        data = json.loads(raw)
        self.assertEqual(data["error"]["type"], "authentication_error")
        self.assertEqual(data["error"]["code"], "invalid_api_key")

    async def test_mock_requires_bearer_auth(self):
        # mock 请求同样经过认证中间件：缺失 token 先被拒绝
        status, _, raw = await self._post(
            {"model": "mock-normal",
             "messages": [{"role": "user", "content": "hi"}]},
            headers={})
        self.assertEqual(status, 401)
        data = json.loads(raw)
        self.assertEqual(data["error"]["type"], "authentication_error")

    async def test_mock_slow_batch_delay_header(self):
        started = time.monotonic()
        status, _, raw = await self._post(
            {"model": "mock-slow-batch",
             "messages": [{"role": "user", "content": "hi"}]},
            headers={"Authorization": "Bearer " + CLIENT_TOKEN,
                     "X-Mock-Response-Delay": "0.2"})
        elapsed = time.monotonic() - started
        self.assertEqual(status, 200)
        data = json.loads(raw)
        self.assertEqual(data["model"], "mock-slow-batch")
        self.assertEqual(data["choices"][0]["message"]["content"],
                         "This is a slow batch response.")
        self.assertGreaterEqual(elapsed, 0.2)
        # 头未生效时会退回默认 5s 延迟，据此断言覆盖生效
        self.assertLess(elapsed, 4.0)

    async def test_mock_stream(self):
        status, headers, raw = await self._post({
            "model": "mock-stream",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(status, 200)
        self.assertIn("text/event-stream",
                      headers.get("Content-Type", ""))
        frames = _parse_sse_frames(raw)
        self.assertEqual(frames[-1], "[DONE]")
        body_frames = frames[:-1]
        # 首帧为 role 通告
        self.assertEqual(body_frames[0]["choices"][0]["delta"]["role"],
                         "assistant")
        contents = [f["choices"][0]["delta"]["content"]
                    for f in body_frames[1:-1]]
        self.assertEqual("".join(contents), "Hello from mock server!")
        self.assertEqual(body_frames[-1]["choices"][0]["finish_reason"],
                         "stop")

    async def test_mock_stream_include_usage(self):
        status, _, raw = await self._post({
            "model": "mock-stream",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "stream_options": {"include_usage": True},
        })
        frames = _parse_sse_frames(raw)
        self.assertEqual(frames[-1], "[DONE]")
        usage_frame = frames[-2]
        self.assertEqual(usage_frame["choices"], [])
        self.assertEqual(usage_frame["usage"],
                         {"prompt_tokens": 10, "completion_tokens": 20,
                          "total_tokens": 30})

    async def test_mock_slow_stream_delay_header(self):
        started = time.monotonic()
        status, _, raw = await self._post(
            {"model": "mock-slow-stream",
             "messages": [{"role": "user", "content": "hi"}],
             "stream": True},
            headers={"Authorization": "Bearer " + CLIENT_TOKEN,
                     "X-Mock-Response-Delay": "0.2"})
        elapsed = time.monotonic() - started
        self.assertEqual(status, 200)
        frames = _parse_sse_frames(raw)
        contents = [f["choices"][0]["delta"]["content"]
                    for f in frames[1:-2]]
        self.assertEqual("".join(contents), "Hello from slow stream!")
        self.assertGreaterEqual(elapsed, 0.2)
        # 头未生效时会退回默认 2s 暂停，据此断言覆盖生效
        self.assertLess(elapsed, 1.5)

    async def test_mock_heartbeat_only(self):
        started = time.monotonic()
        with patch("test.mock_http_server.MOCK_HEARTBEAT_DURATION", 0.5):
            status, _, raw = await self._post({
                "model": "mock-heartbeat",
                "messages": [{"role": "user", "content": "hi"}],
                "stream": True,
            })
        elapsed = time.monotonic() - started
        self.assertEqual(status, 200)
        blocks = [b for b in raw.split(b"\n\n") if b]
        self.assertTrue(blocks)
        # 全部为心跳注释帧，无任何 data 内容帧
        for block in blocks:
            self.assertEqual(block, b": keep-alive")
        self.assertGreaterEqual(elapsed, 0.5)

    async def test_mock_heartbeat_content_interleaved(self):
        status, _, raw = await self._post({
            "model": "mock-heartbeat-content",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
        })
        self.assertEqual(status, 200)
        blocks = [b for b in raw.split(b"\n\n") if b]
        self.assertEqual(blocks[-1], b"data: [DONE]")
        data_blocks = [b for b in blocks if b.startswith(b"data: ")]
        # 心跳注释帧与内容帧交织，且出现在 role 通告帧之后
        self.assertIn(b": keep-alive", blocks)
        self.assertGreater(blocks.index(b": keep-alive"), 0)
        frames = [json.loads(b[len(b"data: "):])
                  for b in data_blocks[:-1]]
        self.assertEqual(frames[0]["choices"][0]["delta"]["role"],
                         "assistant")
        contents = [f["choices"][0]["delta"]["content"]
                    for f in frames[1:-1]]
        self.assertEqual("".join(contents), "Hello from mock server!")
        self.assertEqual(frames[-1]["choices"][0]["finish_reason"], "stop")


# ---------------------------------------------------------------------------
# CLI：http 子命令参数解析、--help 冒烟、run_http 延迟导入分发
# ---------------------------------------------------------------------------

class TestHttpCliSubparser(unittest.TestCase):
    """subparser_http 参数解析：默认值、--client-token append、--help"""

    def _make_parser(self):
        import argparse

        from aitoolman import cli
        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers(dest="subparser_name", required=True)
        cli.subparser_http(subparsers)
        return parser

    def test_defaults(self):
        args = self._make_parser().parse_args(["http"])
        self.assertEqual(args.config, "llm_provider.toml")
        self.assertEqual(args.verbose, 0)
        self.assertIsNone(args.zmq_endpoint)
        self.assertIsNone(args.auth)
        self.assertIsNone(args.host)
        self.assertIsNone(args.port)
        self.assertIsNone(args.client_token)

    def test_all_options_parsed(self):
        args = self._make_parser().parse_args([
            "http",
            "-c", "my.toml",
            "--zmq-endpoint", "tcp://127.0.0.1:5555",
            "--auth", "zmq-secret",
            "--host", "0.0.0.0",
            "--port", "9000",
            "--client-token", "tok-a",
            "--client-token", "tok-b",
        ])
        self.assertEqual(args.config, "my.toml")
        self.assertEqual(args.zmq_endpoint, "tcp://127.0.0.1:5555")
        self.assertEqual(args.auth, "zmq-secret")
        self.assertEqual(args.host, "0.0.0.0")
        self.assertEqual(args.port, 9000)
        self.assertEqual(args.client_token, ["tok-a", "tok-b"])

    def test_help_smoke(self):
        import contextlib
        import io
        parser = self._make_parser()
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as ctx:
                parser.parse_args(["http", "--help"])
        self.assertEqual(ctx.exception.code, 0)
        help_text = buf.getvalue()
        self.assertIn("--client-token", help_text)
        self.assertIn("--zmq-endpoint", help_text)


class TestRunHttpDispatch(unittest.TestCase):
    """run_http 延迟导入并委托 http_server.run_http；main() 按 globals 分发"""

    def test_run_http_delegates_to_http_server(self):
        import argparse

        from aitoolman import cli
        fake_args = argparse.Namespace(config="llm_provider.toml")
        with patch("aitoolman.http_server.run_http") as mock_run:
            cli.run_http(fake_args)
        mock_run.assert_called_once_with(fake_args)

    def test_main_dispatches_http_subcommand(self):
        from aitoolman import cli
        argv = ["aitoolman", "http", "-c", "x.toml"]
        with patch("sys.argv", argv), \
                patch("aitoolman.cli.run_http") as mock_run:
            cli.main()
        args = mock_run.call_args[0][0]
        self.assertEqual(args.config, "x.toml")


# ---------------------------------------------------------------------------
# 全栈集成测试：HTTP 服务 → ZMQ → LLMProviderManager → 上游 mock HTTP 实例
# ---------------------------------------------------------------------------


@unittest.skipUnless(AIOHTTP_AVAILABLE, "aiohttp is required")
class _FullStackTestBase(unittest.IsolatedAsyncioTestCase):
    """全栈集成测试基类：setUpClass 启动上游 mock HTTP 实例，
    并提供 provider 配置构造与 HTTP 服务启停辅助。"""

    @classmethod
    def setUpClass(cls):
        cls.mock_base_url, cls._stop_mock_server = start_mock_http_server(
            CLIENT_TOKEN)

    @classmethod
    def tearDownClass(cls):
        cls._stop_mock_server()

    def _make_provider_config(self):
        """构造指向 mock 服务器的 provider 配置（含正常/慢速/认证失败模型）。"""
        auth = {"Authorization": f"Bearer {CLIENT_TOKEN}"}
        url = self.mock_base_url + "/v1/chat/completions"
        return {
            "default": {
                "timeout": 3.0,
                "timeout_batch": 3.0,
                "max_retries": 0,
                "parallel": 2,
            },
            "api": {
                "mock-normal": {
                    "url": url,
                    "model": "mock-normal",
                    "headers": dict(auth),
                },
                "mock-stream": {
                    "url": url,
                    "model": "mock-stream",
                    "headers": dict(auth),
                },
                "upstream-auth-fail": {
                    "url": url,
                    "model": "mock-normal",
                    "headers": {"Authorization": "Bearer wrong-token"},
                },
                "upstream-slow-stream": {
                    "url": url,
                    "model": "mock-slow-stream",
                    "headers": {**auth, "X-Mock-Response-Delay": "2"},
                },
            },
        }

    def _auth_headers(self):
        return {"Authorization": f"Bearer {CLIENT_TOKEN}"}

    def _chat_body(self, model: str, stream: bool = False):
        body = {"model": model,
                "messages": [{"role": "user", "content": "Hello"}]}
        if stream:
            body["stream"] = True
        return body

    async def _start_http(self, config: HttpServerConfig,
                          provider_config: dict):
        """后台任务运行 serve，并等待 /health 就绪。"""
        self.stop_event = threading.Event()
        self.serve_task = asyncio.create_task(
            serve(config, provider_config, self.stop_event))
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=30))
        deadline = time.monotonic() + 10.0
        while time.monotonic() < deadline:
            try:
                async with self.session.get(
                        self.base_url + "/health",
                        timeout=aiohttp.ClientTimeout(total=0.5)) as resp:
                    if resp.status == 200:
                        return
            except (aiohttp.ClientError, asyncio.TimeoutError, OSError):
                pass
            await asyncio.sleep(0.05)
        self.fail("HTTP server did not become healthy in time")

    async def _stop_http(self):
        """停止 HTTP 服务并校验 serve 正常退出。"""
        await self.session.close()
        self.stop_event.set()
        await asyncio.wait_for(self.serve_task, timeout=15)


class TestFullStackDirectMode(_FullStackTestBase):
    """直连模式全栈：HTTP → inproc（双线程双 loop）→ provider → mock 服务器。"""

    async def asyncSetUp(self):
        self.port = _get_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.provider_config = self._make_provider_config()
        config = build_http_config(self.provider_config, {
            "host": "127.0.0.1",
            "port": self.port,
            "client_tokens": [CLIENT_TOKEN],
        })
        await self._start_http(config, self.provider_config)

    async def asyncTearDown(self):
        await self._stop_http()

    async def test_batch_completion(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("mock-normal"),
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertEqual(resp.headers.get("x-aitoolman-resolved-model"),
                             "mock-normal")
            data = await resp.json()
        self.assertEqual(data["object"], "chat.completion")
        self.assertEqual(data["model"], "mock-normal")
        choice = data["choices"][0]
        self.assertEqual(choice["message"]["role"], "assistant")
        self.assertEqual(choice["message"]["content"],
                         "This is a mock response.")
        self.assertEqual(choice["finish_reason"], "stop")
        self.assertEqual(data["usage"]["prompt_tokens"], 10)
        self.assertEqual(data["usage"]["completion_tokens"], 20)

    async def test_stream_completion(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("mock-stream", stream=True),
        ) as resp:
            self.assertEqual(resp.status, 200)
            self.assertIn("text/event-stream", resp.headers["Content-Type"])
            raw = await resp.read()
        blocks = [b for b in raw.split(b"\n\n") if b]
        self.assertEqual(blocks[-1], b"data: [DONE]")
        frames = [json.loads(block[len(b"data: "):]) for block in blocks[:-1]]
        contents = []
        finish_reasons = []
        for frame in frames:
            self.assertEqual(frame["object"], "chat.completion.chunk")
            choices = frame["choices"]
            if not choices:
                continue
            delta = choices[0]["delta"]
            if delta.get("content"):
                contents.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish_reasons.append(choices[0]["finish_reason"])
        self.assertEqual("".join(contents), "Hello from mock server!")
        self.assertEqual(finish_reasons, ["stop"])

    async def test_upstream_auth_failure_returns_502(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("upstream-auth-fail"),
        ) as resp:
            self.assertEqual(resp.status, 502)
            data = await resp.json()
        self.assertEqual(data["error"]["type"], "upstream_error")

    async def test_unknown_model_returns_404(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("no-such-model"),
        ) as resp:
            self.assertEqual(resp.status, 404)
            data = await resp.json()
        self.assertEqual(data["error"]["type"], "model_not_found")

    async def test_bad_client_token_returns_401(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers={"Authorization": "Bearer wrong-token"},
                json=self._chat_body("mock-normal"),
        ) as resp:
            self.assertEqual(resp.status, 401)

    async def test_client_disconnect_then_server_still_usable(self):
        # 读取部分流式响应后强制断开连接，验证服务端不挂起、后续请求正常
        resp = await self.session.post(
            self.base_url + "/v1/chat/completions",
            headers=self._auth_headers(),
            json=self._chat_body("upstream-slow-stream", stream=True),
        )
        self.assertEqual(resp.status, 200)
        first = await resp.content.readany()
        self.assertTrue(first)
        resp.close()  # 不消费剩余数据，直接断开底层连接
        await asyncio.sleep(0.5)  # 留出断连感知与上游取消的时间
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("mock-normal"),
        ) as resp2:
            self.assertEqual(resp2.status, 200)
            data = await resp2.json()
        self.assertEqual(data["choices"][0]["message"]["content"],
                         "This is a mock response.")

    async def test_list_models(self):
        async with self.session.get(
                self.base_url + "/v1/models",
                headers=self._auth_headers(),
        ) as resp:
            self.assertEqual(resp.status, 200)
            data = await resp.json()
        self.assertEqual(data["object"], "list")
        ids = {m["id"] for m in data["data"]}
        self.assertIn("mock-normal", ids)
        self.assertIn("upstream-slow-stream", ids)


class TestFullStackRemoteMode(_FullStackTestBase):
    """远程模式全栈：HTTP → tcp ZMQ（独立线程 loop 的 LLMZmqServer）→ mock 服务器。"""

    async def asyncSetUp(self):
        self.port = _get_free_port()
        self.base_url = f"http://127.0.0.1:{self.port}"
        zmq_port = _get_free_port()
        internal_token = "internal-" + uuid.uuid4().hex
        server_config = self._make_provider_config()
        server_config["server"] = {
            "zmq_router_rpc": f"tcp://127.0.0.1:{zmq_port}",
            "zmq_auth_token": internal_token,
        }
        self.zmq_server = LLMZmqServer(server_config)
        self.zmq_ready = threading.Event()
        self.zmq_stop = threading.Event()
        self.zmq_thread = threading.Thread(
            target=lambda: asyncio.run(
                self.zmq_server.run(self.zmq_ready, self.zmq_stop)),
            daemon=True,
        )
        self.zmq_thread.start()
        await asyncio.to_thread(self.zmq_ready.wait, 10.0)
        self.assertTrue(self.zmq_ready.is_set())
        self.assertIsNone(self.zmq_server.start_error)

        config = build_http_config({}, {
            "host": "127.0.0.1",
            "port": self.port,
            "zmq_endpoint": f"tcp://127.0.0.1:{zmq_port}",
            "zmq_auth_token": internal_token,
            "client_tokens": [CLIENT_TOKEN],
        })
        await self._start_http(config, {})

    async def asyncTearDown(self):
        await self._stop_http()
        self.zmq_stop.set()
        await asyncio.to_thread(self.zmq_thread.join, 5.0)
        self.assertFalse(self.zmq_thread.is_alive())

    async def test_batch_completion_remote(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("mock-normal"),
        ) as resp:
            self.assertEqual(resp.status, 200)
            data = await resp.json()
        self.assertEqual(data["choices"][0]["message"]["content"],
                         "This is a mock response.")
        self.assertEqual(data["choices"][0]["finish_reason"], "stop")
        self.assertEqual(data["usage"]["prompt_tokens"], 10)
        self.assertEqual(data["usage"]["completion_tokens"], 20)

    async def test_stream_completion_remote(self):
        async with self.session.post(
                self.base_url + "/v1/chat/completions",
                headers=self._auth_headers(),
                json=self._chat_body("mock-stream", stream=True),
        ) as resp:
            self.assertEqual(resp.status, 200)
            raw = await resp.read()
        blocks = [b for b in raw.split(b"\n\n") if b]
        self.assertEqual(blocks[-1], b"data: [DONE]")
        frames = [json.loads(block[len(b"data: "):]) for block in blocks[:-1]]
        contents = []
        finish_reasons = []
        for frame in frames:
            choices = frame["choices"]
            if not choices:
                continue
            delta = choices[0]["delta"]
            if delta.get("content"):
                contents.append(delta["content"])
            if choices[0].get("finish_reason"):
                finish_reasons.append(choices[0]["finish_reason"])
        self.assertEqual("".join(contents), "Hello from mock server!")
        self.assertEqual(finish_reasons, ["stop"])


if __name__ == "__main__":
    unittest.main()
