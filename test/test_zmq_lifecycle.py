"""ZMQ 传输层生命周期测试。

覆盖服务端/客户端配套修改的行为契约：
- Context 注入与所有权：外部注入的 ctx 在 close 时不被 term；自建 ctx 在 close 时 term
- initialize() 幂等：重复调用不重复绑定端点
- run(on_ready, stop_event) 优雅停止：stop_event 置位后 run 在 RCVTIMEO 轮询内返回
- 初始化失败经 start_error 传播，run 正常返回
- inproc 双线程双 loop smoke：验证跨线程共享 zmq.asyncio.Context 的可行性
"""

import asyncio
import threading
import unittest
import uuid

import zmq
import zmq.asyncio

from test.mock_http_server import start_mock_http_server

from aitoolman.model import Message
from aitoolman.zmqclient import LLMZmqClient
from aitoolman.zmqserver import LLMZmqServer

CLIENT_TOKEN = "test-secret-token"


def _make_server_config(router_rpc: str, auth_token: str = "internal-token"):
    """构建最小可用的服务端配置（server 节 + 空的 provider 配置节）"""
    return {
        "server": {
            "zmq_router_rpc": router_rpc,
            "zmq_auth_token": auth_token,
        },
        "default": {},
        "api": {},
    }


class TestServerContextOwnership(unittest.IsolatedAsyncioTestCase):
    """服务端 Context 所有权：外部注入的 ctx close 时不 term，自建 ctx close 时 term"""

    async def test_external_ctx_not_term_on_close(self):
        ctx = zmq.asyncio.Context()
        try:
            server = LLMZmqServer(
                _make_server_config(f"inproc://own-external-{uuid.uuid4().hex}"),
                ctx=ctx)
            self.assertFalse(server._owns_ctx)
            await server.close()
            self.assertFalse(ctx.closed)
        finally:
            ctx.term()

    async def test_owned_ctx_term_on_close(self):
        server = LLMZmqServer(
            _make_server_config(f"inproc://own-internal-{uuid.uuid4().hex}"))
        self.assertTrue(server._owns_ctx)
        owned_ctx = server.ctx
        await server.close()
        self.assertTrue(owned_ctx.closed)


class TestClientContextOwnership(unittest.IsolatedAsyncioTestCase):
    """客户端 Context 所有权：外部注入的 ctx close 时不 term，自建 ctx close 时 term"""

    async def test_external_ctx_not_term_on_close(self):
        ctx = zmq.asyncio.Context()
        try:
            client = LLMZmqClient("tcp://127.0.0.1:5555", "token", ctx=ctx)
            self.assertFalse(client._owns_ctx)
            client.connected = True  # 模拟已连接以进入 close 清理路径
            await client.close()
            self.assertFalse(ctx.closed)
        finally:
            ctx.term()

    async def test_owned_ctx_term_on_close(self):
        client = LLMZmqClient("tcp://127.0.0.1:5555", "token")
        self.assertTrue(client._owns_ctx)
        owned_ctx = client.ctx
        client.connected = True
        await client.close()
        self.assertTrue(owned_ctx.closed)


class TestServerInitializeIdempotent(unittest.IsolatedAsyncioTestCase):
    """initialize() 幂等：重复调用不重复绑定端点"""

    async def test_repeated_initialize_does_not_rebind(self):
        ctx = zmq.asyncio.Context()
        server = LLMZmqServer(
            _make_server_config(f"inproc://init-idempotent-{uuid.uuid4().hex}"),
            ctx=ctx)
        try:
            await server.initialize()
            # 第二次调用若不幂等会因重复 bind 抛 zmq.ZMQError
            await server.initialize()
            self.assertTrue(server._initialized)
        finally:
            await server.close()
            ctx.term()


class TestServerGracefulStop(unittest.IsolatedAsyncioTestCase):
    """run(on_ready, stop_event) 优雅停止"""

    async def test_stop_event_returns_run_promptly(self):
        server = LLMZmqServer(
            _make_server_config(f"inproc://graceful-{uuid.uuid4().hex}"))
        on_ready = threading.Event()
        stop_event = threading.Event()
        task = asyncio.create_task(server.run(on_ready, stop_event))
        try:
            # 经 to_thread 等待就绪事件，避免阻塞事件循环
            await asyncio.wait_for(asyncio.to_thread(on_ready.wait, 5.0), timeout=6.0)
            self.assertTrue(on_ready.is_set())
            self.assertIsNone(server.start_error)

            stop_event.set()
            # RCVTIMEO=100ms 轮询下 run 应在宽限时间内返回
            await asyncio.wait_for(task, timeout=2.0)
        finally:
            stop_event.set()
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass


class TestServerInitFailure(unittest.IsolatedAsyncioTestCase):
    """初始化失败经 start_error 传播，run 正常返回"""

    async def test_invalid_endpoint_reports_start_error(self):
        server = LLMZmqServer(_make_server_config("bogus://invalid-endpoint"))
        on_ready = threading.Event()
        stop_event = threading.Event()
        # run 捕获初始化异常后应正常返回而非抛出
        await asyncio.wait_for(server.run(on_ready, stop_event), timeout=5.0)
        self.assertIsNotNone(server.start_error)
        self.assertFalse(on_ready.is_set())


class TestInprocCrossThreadSmoke(unittest.IsolatedAsyncioTestCase):
    """inproc 双线程双 loop smoke：后台线程 LLMZmqServer + 主线程 LLMZmqClient 共享 Context"""

    @classmethod
    def setUpClass(cls):
        cls.base_url, cls._stop_mock_server = start_mock_http_server(CLIENT_TOKEN)

    @classmethod
    def tearDownClass(cls):
        cls._stop_mock_server()

    async def test_batch_request_across_threads(self):
        ctx = zmq.asyncio.Context()
        internal_token = "internal-" + uuid.uuid4().hex
        endpoint = f"inproc://http-smoke-{uuid.uuid4().hex}"
        config = {
            "server": {
                "zmq_router_rpc": endpoint,
                "zmq_auth_token": internal_token,
            },
            "default": {},
            "api": {
                "mock-normal": {
                    "url": self.base_url + "/v1/chat/completions",
                    "model": "mock-normal",
                    "headers": {"Authorization": f"Bearer {CLIENT_TOKEN}"},
                },
            },
        }
        server = LLMZmqServer(config, ctx=ctx)
        on_ready = threading.Event()
        stop_event = threading.Event()
        thread = threading.Thread(
            target=lambda: asyncio.run(server.run(on_ready, stop_event)),
            daemon=True,
        )
        thread.start()
        try:
            self.assertTrue(
                on_ready.wait(timeout=10),
                f"server thread failed to start: {server.start_error}")
            self.assertIsNone(server.start_error)

            client = LLMZmqClient(endpoint, internal_token, ctx=ctx)
            await client.initialize()
            try:
                request = await client.request(
                    "mock-normal",
                    [Message(role="user", content="Hello")])
                response = await asyncio.wait_for(request.response, timeout=10)
                self.assertEqual(response.finish_reason, "stop")
                self.assertIsNone(response.error_text)
                self.assertEqual(response.response_text, "This is a mock response.")
                self.assertEqual(response.prompt_tokens, 10)
                self.assertEqual(response.completion_tokens, 20)
            finally:
                await client.close()
        finally:
            stop_event.set()
            thread.join(timeout=5)
            self.assertFalse(thread.is_alive(), "server thread did not stop in time")
            ctx.term()


if __name__ == "__main__":
    unittest.main()
