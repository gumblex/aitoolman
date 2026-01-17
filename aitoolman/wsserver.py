"""
WebSocket 服务器模块 - 为 aitoolman 框架提供 WebSocket 交互后端
使用 websockets 库，支持实时双向通信
"""
import abc
import json
import asyncio
import logging
import urllib.parse
from abc import ABC
from dataclasses import dataclass
from typing import Any, Dict, Optional, Callable
from enum import Enum

import websockets
from websockets.exceptions import ConnectionClosedOK, ConnectionClosedError

from . import util
from . import app as _app
from . import client as _client
from . import channel as _channel

logger = logging.getLogger(__name__)

class WSEventType(str, Enum):
    """WebSocket 事件类型"""
    REQUEST = "request"  # 客户端请求调用处理器
    RESPONSE = "response"  # 服务器返回最终结果
    CHANNEL_INPUT = "channel_input"  # 客户端向通道写入数据
    CHANNEL_OUTPUT = "channel_output"  # 服务器向客户端写入通道数据
    ERROR = "error"  # 错误消息
    PING = "ping"  # 心跳检测
    PONG = "pong"  # 心跳响应
    READY = "ready"  # 服务器就绪事件


@dataclass
class WSEvent:
    """WebSocket 事件数据结构"""
    type: WSEventType
    data: Dict[str, Any]
    request_id: Optional[str] = None  # 关联的请求ID


def get_client_ip(
        conn: websockets.ServerConnection,
        header: Optional[str] = None
) -> str:
    """
    获取 WebSocket 客户端的 IP 地址

    Args:
        conn: WebSocket 连接对象
        header: 真实 IP 头

    Returns:
        客户端 IP 地址字符串
    """
    if header:
        if header.lower() == 'x-forwarded-for':
            header_value = conn.request.headers.get('X-Forwarded-For')
            try:
                # X-Forwarded-For 可能包含多个 IP，用逗号分隔
                # 第一个是最原始的客户端 IP
                first_ip = header_value.split(',')[0].strip()
                if first_ip:
                    return first_ip
            except Exception:
                # 如果解析失败，继续下一个
                pass
        else:
            header_value = conn.request.headers.get(header)
            if header_value and header_value.strip():
                return header_value.strip()
    return conn.remote_address[0]


class WebsocketChannel(_channel.TextChannel):
    """
    适配 WebSocket 的文本通道
    重写 read/write 方法，直接通过 WebSocket 收发信息
    """
    def __init__(
            self,
            ws_handler: 'BaseWsSession',
            channel_name: str,
            read_fragments: bool = False
    ):
        super().__init__(read_fragments=read_fragments)
        self.ws_handler = ws_handler
        self.channel_name = channel_name

    async def write_message(self, message: Optional[str]):
        """重写：通过 WebSocket 发送完整消息"""
        logger.debug(f"Channel {self.channel_name} write_message: {message}")
        if self.closed:
            raise IOError("Channel is closed")

        event = WSEvent(
            type=WSEventType.CHANNEL_OUTPUT,
            data={
                "channel": self.channel_name,
                "mode": "message",
                "data": message,
                "end": False
            }
        )
        # 发送给所有连接的客户端
        tasks = []
        for client in self.clients:
            task = asyncio.create_task(
                self.ws_handler.send_event(client, event)
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

    async def write_fragment(self, text: str, end: bool = False):
        """重写：通过 WebSocket 发送消息片段"""
        logger.debug(
            f"Channel {self.channel_name} write_fragment: {text[:50]}..., end={end}")
        if self.closed:
            raise IOError("Channel is closed")

        event = WSEvent(
            type=WSEventType.CHANNEL_OUTPUT,
            data={
                "channel": self.channel_name,
                "mode": "fragment",
                "data": text,
                "end": end
            }
        )
        # 发送给所有连接的客户端
        tasks = []
        for client in self.clients:
            task = asyncio.create_task(
                self.ws_handler.send_event(client, event)
            )
            tasks.append(task)
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)

class BaseWsSession(ABC):
    """
    WebSocket LLM 应用基类
    开发者应继承此类并实现相关方法
    """
    handlers: Dict[str, Callable] = {}

    def __init__(self, config: Dict[str, Any], client: _client.LLMClient):
        """
        初始化 WebSocket LLM 应用
        Args:
            config: LLM 应用配置
            client: LLM 客户端实例（由服务器共享）
        """
        self.config = config
        self.client: _client.LLMClient = client
        self.context_id = util.get_id()
        self.app: Optional[_app.LLMApplication] = None
        self.channels: Dict[str, WebsocketChannel] = {}
        self.active_requests: Dict[str, asyncio.Task] = {}
        self.ready = False

        # 注册默认通道
        self._register_default_channels()

    def _register_default_channels(self):
        """注册默认通道"""
        # stdin: 标准输入通道
        self.channels['stdin'] = WebsocketChannel(
            self, 'stdin', read_fragments=False
        )
        # stdout: 标准输出通道
        self.channels['stdout'] = WebsocketChannel(
            self, 'stdout', read_fragments=True
        )
        # reasoning: 推理通道
        self.channels['reasoning'] = WebsocketChannel(
            self, 'reasoning', read_fragments=True
        )

    @classmethod
    def handler(cls, name: str):
        """
        装饰器：注册请求处理器
        Args:
            name: 处理器名称
        """
        def decorator(func: Callable):
            cls.handlers[name] = func
            return func
        return decorator

    async def initialize_session(self):
        """初始化应用（创建 LLMApplication）"""
        self.app = _app.LLMApplication(
            client=self.client,
            config_dict=self.config,
            channels=self.channels,
            context_id=self.context_id
        )

        # 初始化所有模块
        self.app.init_all_modules()
        logger.info(
            "[%s] BaseWsSession initialized.", self.context_id
        )

    async def cleanup_session(self):
        """清理资源"""
        # 取消所有活跃请求
        for request_id, task in self.active_requests.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
        self.active_requests.clear()
        logger.info("[%s] BaseWsSession closed.", self.context_id)

    @abc.abstractmethod
    async def authenticate(self, token: Optional[str]) -> bool:
        return True

    # ========== 事件回调函数（开发者可重写） ==========
    async def on_connect(self, conn: websockets.ServerConnection, path: str):
        """
        连接建立时的回调（在认证成功后调用）
        Args:
            conn: WebSocket 连接
            path: 请求路径
        """
        logger.info(
            "[%s] Client connected: %s, path: %s",
            self.context_id, conn.remote_address, path
        )

    async def on_disconnect(self, conn: websockets.ServerConnection):
        """
        连接断开时的回调
        Args:
            conn: WebSocket 连接
        """
        logger.info(f"Client disconnected: {conn.remote_address}")

    async def on_channel_input(
            self,
            conn: websockets.ServerConnection,
            channel_name: str,
            text: str
    ):
        """
        接收到通道数据时的回调
        Args:
            conn: WebSocket 连接
            channel_name: 通道名称
            text: 文本内容
        """
        logger.debug(
            f"Channel read: {channel_name}, text: {text[:50]}...")

    async def on_error(
            self,
            conn: websockets.ServerConnection,
            error: Exception
    ):
        """
        发生错误时的回调
        Args:
            conn: WebSocket 连接
            error: 异常对象
        """
        logger.error(f"WebSocket error: {error}", exc_info=True)

    # ========== 内部处理方法 ==========
    async def send_event(
            self,
            websocket: websockets.ServerConnection,
            event: WSEvent
    ) -> bool:
        """
        发送事件到 WebSocket
        Args:
            websocket: WebSocket 连接
            event: 事件对象
        Returns:
            bool: 是否发送成功
        """
        try:
            event_dict = {
                "type": event.type.value,
                "data": event.data
            }
            if event.request_id:
                event_dict["request_id"] = event.request_id
            await websocket.send(json.dumps(event_dict))
            return True
        except (ConnectionClosedOK, ConnectionClosedError):
            logger.debug(f"Connection closed while sending event: {event.type}")
            return False
        except Exception as e:
            logger.error(f"Failed to send event: {e}")
            return False

    async def send_error(
            self,
            websocket: websockets.ServerConnection,
            error_msg: str,
            request_id: Optional[str] = None
    ):
        """
        发送错误消息
        Args:
            websocket: WebSocket 连接
            error_msg: 错误信息
            request_id: 关联的请求ID
        """
        event = WSEvent(
            type=WSEventType.ERROR,
            data={"message": error_msg},
            request_id=request_id
        )
        await self.send_event(websocket, event)

    async def send_response(
            self,
            websocket: websockets.ServerConnection,
            request_id: str,
            data: Any
            ):
        """
        发送响应消息
        Args:
            websocket: WebSocket 连接
            request_id: 请求ID
            data: 响应数据
        """
        event = WSEvent(
            type=WSEventType.RESPONSE,
            data=data,
            request_id=request_id
        )
        await self.send_event(websocket, event)

    async def _handle_request(
            self,
            conn: websockets.ServerConnection,
            request_id: str,
            handler_name: str,
            kwargs: Dict[str, Any]
    ):
        """
        内部方法：分发请求到注册的处理器
        Args:
            conn: WebSocket 连接
            request_id: 请求ID
            handler_name: 处理器名称
            kwargs: 请求参数
        """
        if handler_name not in self.handlers:
            await self.send_error(conn, f"Handler not found: {handler_name}", request_id)
            return

        try:
            # 调用注册的处理器
            result = await self.handlers[handler_name](self, conn, kwargs)
            await self.send_response(conn, request_id, result)
        except Exception as e:
            logger.error(f"Handler {handler_name} error: {e}", exc_info=True)
            await self.send_error(conn, f"Handler error: {str(e)}", request_id)
        finally:
            try:
                self.active_requests[request_id]
            except KeyError:
                pass

    async def handle_message(
            self,
            conn: websockets.ServerConnection,
            data: Dict[str, Any]
    ):
        """
        处理接收到的消息
        Args:
            conn: WebSocket 连接
            message: 消息内容
        """
        event_type = data.get("type")

        if event_type == WSEventType.REQUEST.value:
            # 处理请求
            request_id = data.get("request_id")
            if not request_id:
                await self.send_error(conn, "Missing request_id")
                return

            if not self.ready:
                await self.send_error(conn, "Not ready", request_id)
                return

            handler_name = data.get("handler")
            kwargs = data.get("kwargs", {})

            if not handler_name:
                await self.send_error(conn, "Missing handler name", request_id)
                return

            # 创建异步任务处理请求
            task = asyncio.create_task(
                self._handle_request(conn, request_id, handler_name, kwargs)
            )
            self.active_requests[request_id] = task

        elif event_type == WSEventType.CHANNEL_INPUT.value:
            # 处理通道读取
            channel_name = data.get("channel")
            data = data.get("data", "")

            if channel_name and channel_name in self.channels:
                # 调用开发者回调
                await self.on_channel_input(conn, channel_name, data)
                # 写入通道
                channel = self.channels[channel_name]
                await channel.write_message(data)
            else:
                await self.send_error(conn, f"Channel not found: {channel_name}")

        elif event_type == WSEventType.PING.value:
            # 心跳响应
            event = WSEvent(
                type=WSEventType.PONG,
                data={"timestamp": data.get("timestamp")}
            )
            await self.send_event(conn, event)

        else:
            await self.send_error(conn, f"Unknown event type: {event_type}")

    async def handle_connection(
            self, conn: websockets.ServerConnection, path: str
    ):
        """
        处理单个 WebSocket 连接
        Args:
            conn: WebSocket 连接
            path: 请求路径
        """
        # 从 URL 参数读取 token
        query_params = urllib.parse.parse_qs(urllib.parse.urlparse(path).query)
        token = query_params.get('token', [None])[0]

        # 认证
        if not await self.authenticate(token):
            await self.send_error(conn, "Authentication failed")
            await conn.close(code=1008, reason="Unauthorized")
            return

        # 初始化应用（创建 LLMApplication）
        await self.initialize_session()

        # 连接建立回调（开发者可在此读取数据库等）
        await self.on_connect(conn, path)

        # 标记为就绪并发送 ready 事件
        self.ready = True
        await self.send_event(conn, WSEvent(type=WSEventType.READY, data={}))

        try:
            # 主消息循环
            async for message in conn:
                try:
                    data = json.loads(message)
                except json.JSONDecodeError:
                    await self.send_error(conn, "Invalid JSON format")
                    continue
                try:
                    await self.handle_message(conn, data)
                except Exception as e:
                    error_text = f"Handle message error: {type(e).__name__}: {str(e)}"
                    logger.exception("[%s] %s", self.context_id, error_text)
                    await self.send_error(conn, error_text)

        except (ConnectionClosedOK, ConnectionClosedError):
            logger.debug(
                f"Connection closed normally: {conn.remote_address}")
        except Exception as e:
            logger.error(f"Connection error: {e}", exc_info=True)
            await self.on_error(conn, e)
        finally:
            # 清理
            self.ready = False
            await self.on_disconnect(conn)
            await self.cleanup_session()

class WebsocketServer:
    """
    WebSocket 服务器
    管理多个 WebSocket 连接和 LLM 客户端
    """
    def __init__(
            self,
            client: _client.LLMClient,
            host: str = "localhost",
            port: int = 8765,
            ssl_context=None
    ):
        """
        初始化 WebSocket 服务器
        Args:
            client: LLM 客户端实例（由服务器管理）
            host: 服务器主机
            port: 服务器端口
            ssl_context: SSL 上下文（用于 wss）
        """
        self.client = client
        self.host = host
        self.port = port
        self.ssl_context = ssl_context
        self.server = None
        self.path_handlers: Dict[
            str, Callable[[_client.LLMClient], BaseWsSession]] = {}
        logger.info(f"WebsocketServer initialized: {host}:{port}")

    def register_app(
            self,
            path: str,
            app_factory: Callable[[_client.LLMClient], BaseWsSession]
    ):
        """
        注册应用工厂
        Args:
            path: WebSocket 路径
            app_factory: 应用工厂函数，接收 client 参数
        """
        self.path_handlers[path] = app_factory
        logger.info(f"Registered app for path: {path}")

    async def _connection_handler(
        self, conn: websockets.ServerConnection, path: str
    ):
        """
        连接处理函数
        Args:
            conn: WebSocket 连接
            path: 请求路径
        """
        # 根据路径选择应用
        if path not in self.path_handlers:
            await conn.close(code=1003, reason=f"Unsupported path: {path}")
            return

        app_factory = self.path_handlers[path]
        app_instance = app_factory(self.client)
        app_instance.context_id = util.get_host_id(get_client_ip(conn))

        # 处理连接
        await app_instance.handle_connection(conn, path)

    async def start(self):
        """
        启动 WebSocket 服务器
        """
        try:
            # 初始化客户端
            await self.client.__aenter__()

            self.server = await websockets.serve(
                self._connection_handler,
                self.host,
                self.port,
                ssl=self.ssl_context
            )
            logger.info(
                f"WebSocket server started on ws://{self.host}:{self.port}")
            logger.info(f"Registered paths: {list(self.path_handlers.keys())}")

            # 保持服务器运行
            await self.server.wait_closed()
        except (KeyboardInterrupt, asyncio.CancelledError):
            await self.stop()
        except Exception as e:
            logger.error(f"Failed to start WebSocket server: {e}")
            raise

    async def stop(self):
        """
        停止 WebSocket 服务器
        """
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            logger.info("WebSocket server stopped")

        # 清理客户端
        await self.client.__aexit__(None, None, None)

# ========== 使用示例 ==========
class DemoWebsocketApp(BaseWsSession):
    """示例 WebSocket 应用"""

    @BaseWsSession.handler("chat")
    async def handle_chat(self, conn: websockets.ServerConnection, kwargs: Dict[str, Any]):
        """处理聊天请求"""
        user_input = kwargs.get("message", "")
        if not user_input:
            raise ValueError("Message is required")

        # 调用 LLM 模块
        result = await self.app.user_input(user_input=user_input)
        return {
            "text": result.text,
            "status": result.status.value,
            "data": result.data
        }

    @BaseWsSession.handler("echo")
    async def handle_echo(self, conn: websockets.ServerConnection, kwargs: Dict[str, Any]):
        """处理回声请求"""
        return {"echo": kwargs.get("message", "")}

    async def authenticate(self, token: Optional[str]) -> bool:
        """示例认证（实际应实现 JWT 验证）"""
        # 示例：简单 token 验证
        return token == "demo-token"

    async def on_connect(self, conn: websockets.ServerConnection, path: str):
        """连接建立（可在此读取数据库等）"""
        await super().on_connect(conn, path)
        logger.info("Demo app connected, ready to handle requests")


async def run_demo_server():
    """运行示例服务器"""
    import os
    # 加载配置
    config_path = os.path.join(os.path.dirname(__file__), "llm_config.toml")
    config = util.load_config(config_path)

    # 创建共享的 LLM 客户端
    client = _client.LLMLocalClient(config)

    # 创建服务器
    server = WebsocketServer(client=client, host="0.0.0.0", port=8765)

    # 注册应用工厂
    def create_demo_app(client: _client.LLMClient):
        return DemoWebsocketApp(config, client)

    server.register_app("/llm", create_demo_app)

    # 启动服务器
    await server.start()


if __name__ == "__main__":
    # 运行示例服务器
    asyncio.run(run_demo_server())
