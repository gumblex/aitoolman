"""测试共享辅助：以 HTTP 服务端内置 mock 模式作为测试上游。

启动一个真实的 aitoolman HTTP 服务端实例（mock=True），测试经与生产一致
的服务路径访问 mock 场景（mock-normal、mock-stream、mock-slow-batch、
mock-slow-stream、mock-heartbeat、mock-heartbeat-content、mock-auth-error），
无需任何真实上游模型配置。

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
from typing import Callable, Tuple

from aitoolman.http_server import build_http_config, serve

# 就绪等待与停止等待的超时时间（秒）
READY_TIMEOUT = 10.0
STOP_TIMEOUT = 10.0


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
    """在随机空闲端口启动 mock 模式 HTTP 服务端并等待就绪。

    Args:
        client_token: 客户端访问所需的 Bearer 令牌；服务端以 mock=True
            启动，mock 请求在 resolve_model 之前拦截，无需真实模型配置。
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
        "mock": True,
        "client_tokens": [client_token],
    })
    provider_config = {"server": {}, "default": {}, "api": {}}
    stop_event = threading.Event()
    thread = threading.Thread(
        target=lambda: asyncio.run(serve(config, provider_config, stop_event)),
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
