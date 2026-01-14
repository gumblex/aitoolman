import json
import time
import asyncio
import logging
import sqlite3
from typing import Optional, List, Dict, Any

import zmq
import zmq.asyncio

from . import util
from .model import LLMRequest, LLMResponse, FinishReason, ToolCall, Message
from .channel import TextChannel
from .client import LLMClient

logger = logging.getLogger(__name__)


class LLMZmqClient(LLMClient):
    def __init__(self, router_endpoint: str):
        super().__init__()
        self.router_endpoint = router_endpoint
        self.ctx = zmq.asyncio.Context()
        self.socket = self.ctx.socket(zmq.DEALER)
        self.socket.setsockopt_string(zmq.IDENTITY, self.client_id)
        self.active_requests: Dict[str, LLMRequest] = {}  # request_id -> LLMRequest
        self.listener_task: Optional[asyncio.Task] = None
        self.connected = False

    async def connect(self):
        """连接服务器并启动监听"""
        if self.connected:
            return
        self.socket.connect(self.router_endpoint)
        self.listener_task = asyncio.create_task(self.listen_responses())
        self.connected = True
        logger.info(f"Connected to {self.router_endpoint}")

    async def initialize(self):
        await self.connect()

    async def close(self):
        """关闭客户端"""
        if not self.connected:
            return
        # 取消所有活跃请求
        for request_id in list(self.active_requests.keys()):
            await self.cancel(request_id)
        # 停止监听
        if self.listener_task:
            self.listener_task.cancel()
            try:
                await self.listener_task
            except asyncio.CancelledError:
                pass
        # 清理资源
        self.socket.close()
        self.ctx.term()
        self.connected = False
        logger.info("Client closed")

    async def listen_responses(self):
        """监听服务器响应"""
        while self.connected:
            try:
                message = await self.socket.recv_multipart()
                if len(message) != 2:
                    logger.error(f"Invalid response format: {len(message)} parts")
                    continue

                json_data = json.loads(message[1].decode('utf-8'))
                msg_type = json_data.get('type')
                request_id = json_data.get('request_id')

                request = self.active_requests.get(request_id)
                if not request:
                    logger.warning(f"Response for unknown request {request_id}")
                    continue

                if msg_type == 'channel_write':
                    await self.handle_channel_write(request, json_data)
                elif msg_type == 'response':
                    await self.handle_response(request, json_data)
                elif msg_type == 'cancel_ack':
                    await self.handle_cancel_ack(request)
                else:
                    logger.warning(f"Unknown response type: {msg_type}")
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.exception("Error in listener task")

    async def handle_channel_write(self, request: LLMRequest, json_data: Dict[str, Any]):
        """处理channel写入消息"""
        channel_type = json_data['channel']
        mode = json_data['mode']
        text = json_data['text']
        end = json_data['end']

        channel = None
        if channel_type == 'response':
            channel = request.response_channel
        elif channel_type == 'reasoning':
            channel = request.reasoning_channel

        if not channel:
            logger.debug(f"Request {request.request_id} has no {channel_type} channel")
            return

        if mode == 'fragment':
            await channel.write_fragment(text, end)
        elif mode == 'message':
            await channel.write_message(text)

    async def handle_response(self, request: LLMRequest, json_data: Dict[str, Any]):
        """处理完整响应"""
        response_data = json_data['response']
        # 构造LLMResponse
        finish_reason = FinishReason(response_data['finish_reason']) if response_data['finish_reason'] in FinishReason.__members__ else response_data['finish_reason']
        response = LLMResponse(
            client_id=response_data['client_id'],
            context_id=response_data['context_id'],
            request_id=response_data['request_id'],
            model_name=response_data['model_name'],
            stream=response_data['stream'],
            start_time=response_data['start_time'],
            queue_time=response_data['queue_time'],
            queue_length=response_data['queue_length'],
            time_to_first_token=response_data['time_to_first_token'],
            total_response_time=response_data['total_response_time'],
            response_text=response_data['response_text'],
            response_reasoning=response_data['response_reasoning'],
            response_tool_calls=[ToolCall(**tc) for tc in (response_data['response_tool_calls'] or [])],
            finish_reason=finish_reason,
            error_text=response_data['error_text'],
            prompt_tokens=response_data['prompt_tokens'],
            completion_tokens=response_data['completion_tokens'],
            response_message=response_data['response_message']
        )
        request.response.set_result(response)
        del self.active_requests[request.request_id]

    async def handle_cancel_ack(self, request: LLMRequest):
        """处理取消确认"""
        request.is_cancelled = True
        del self.active_requests[request.request_id]
        logger.info("[%s] Request cancelled", request.request_id)

    async def request(
            self,
            model_name: str,
            messages: List[Message],
            tools: Dict[str, Dict[str, Any]] = None,
            options: Optional[Dict[str, Any]] = None,
            stream: bool = False,
            context_id: Optional[str] = None,
            response_channel: Optional[TextChannel] = None,
            reasoning_channel: Optional[TextChannel] = None
    ) -> LLMRequest:
        """发送LLM请求（实现LLMClient抽象方法）"""
        if not self.connected:
            raise RuntimeError("Client not connected")

        # 构造LLMRequest
        request = self._make_request(
            model_name, messages, tools, options, stream,
            context_id, response_channel, reasoning_channel
        )
        self.active_requests[request.request_id] = request

        # 发送请求消息
        request_msg = {
            'type': 'request',
            'client_id': request.client_id,
            'context_id': request.context_id,
            'request_id': request.request_id,
            'model_name': request.model_name,
            'messages': [m.to_dict() for m in request.messages],
            'tools': request.tools,
            'options': request.options,
            'stream': request.stream
        }
        await self.socket.send_multipart([
            b'',
            util.encode_message(request_msg)
        ])
        return request

    async def cancel(self, request_id: str):
        """取消指定请求（实现LLMClient抽象方法）"""
        if not self.connected:
            raise RuntimeError("Client not connected")

        request = self.active_requests.get(request_id)
        if not request:
            logger.warning("[%s] Request not found", request_id)
            return

        # 发送取消消息
        cancel_msg = {
            'type': 'cancel',
            'client_id': self.client_id,
            'request_id': request_id
        }
        await self.socket.send_multipart([
            b'',
            util.encode_message(cancel_msg)
        ])

    async def cancel_all(self, context_id: Optional[str] = None):
        """取消当前客户端的所有请求"""
        if not self.connected:
            raise RuntimeError("Client not connected")

        cancel_msg = {
            'type': 'cancel_all',
            'client_id': self.client_id,
            'context_id': context_id
        }
        await self.socket.send_multipart([
            b'',
            util.encode_message(cancel_msg)
        ])

    async def audit_event(self, context_id: str, event_type: str, **kwargs):
        if not self.connected:
            raise RuntimeError("Client not connected")

        audit_msg = {
            'type': 'audit_event',
            'client_id': self.client_id,
            'context_id': context_id,
            'event_type': event_type,
            'data': kwargs,
            'timestamp': time.time()
        }
        await self.socket.send_multipart([
            b'',
            util.encode_message(audit_msg)
        ])


class LLMMonitor:
    def __init__(self, pub_endpoint: str):
        self.pub_endpoint = pub_endpoint
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.running = False

    def start(self):
        """开始监听审计消息"""
        self.socket.connect(self.pub_endpoint)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "llm_request")
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "audit_event")
        self.running = True
        while self.running:
            try:
                message = self.socket.recv_multipart()
                if len(message) >= 2:
                    topic = message[0].decode('utf-8')
                    if topic == "llm_request":
                        data = json.loads(message[1].decode('utf-8'))
                        self.on_llm_request(data)
                    elif topic == "audit_event":
                        data = json.loads(message[1].decode('utf-8'))
                        self.on_audit_event(data)
            except zmq.ZMQError as e:
                if self.running:
                    logger.info(f"Monitor ZMQ error: {e}")
            except Exception as e:
                logger.info(f"Monitor error: {e}")

    def on_llm_request(self, data: Dict[str, Any]):
        """处理审计消息（子类可重写）"""
        # 使用新的字段名
        logger.info(
            "[AUDIT LLM: %s/%s] model: %s, queue: %.1f s, first_token: %.1f s, total: %.1f s, tokens: %s/%s, reason: %s",
            data['client_id'],
            data['request_id'],
            data['model_name'],
            data.get('queue_time', 0) or 0,
            data.get('time_to_first_token', 0) or 0,
            data.get('total_response_time', 0) or 0,
            data.get('prompt_tokens', 0),
            data.get('completion_tokens', 0),
            data.get('finish_reason', 'unknown')
        )

    def on_audit_event(self, data: Dict[str, Any]):
        """处理审计事件消息（子类可重写）"""
        logger.info(
            "[AUDIT EVENT: %s/%s] event: %s, %s",
            data['client_id'], data['context_id'],
            data['event_type'], data['kwargs']
        )

    def stop(self):
        """停止监听"""
        self.running = False
        self.socket.close()
        self.ctx.term()


class DBLLMMonitor(LLMMonitor):
    def __init__(self, pub_endpoint: str, db_path: str = "llm_audit.db"):
        super().__init__(pub_endpoint)
        self.db_path = db_path
        self._init_database()

    def _init_database(self):
        """初始化数据库表，更新字段以匹配新的数据结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS llm_requests (
                id INTEGER PRIMARY KEY,
                client_id TEXT,
                context_id TEXT,
                request_id TEXT UNIQUE,
                model_name TEXT,
                stream INTEGER,
                start_time REAL,
                queue_time REAL,
                queue_length INTEGER,
                time_to_first_token REAL,
                total_response_time REAL,
                response_text TEXT,
                response_reasoning TEXT,
                response_tool_calls TEXT,
                finish_reason TEXT,
                error_text TEXT,
                prompt_tokens INTEGER,
                completion_tokens INTEGER,
                request_messages TEXT,
                request_tools TEXT,
                request_options TEXT,
                response_message TEXT,
                created_at INTEGER
            )
        ''')
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY,
                client_id TEXT,
                context_id TEXT,
                event_type TEXT,
                data TEXT,
                timestamp REAL
            )
        ''')
        conn.commit()
        conn.close()

    def on_llm_request(self, data: Dict[str, Any]):
        """将审计消息存入数据库，使用新的字段结构"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        db_data = data.copy()
        db_data['created_at'] = int(time.time())
        keys, qs, values = util.make_insert_auto(db_data)
        cursor.execute('INSERT OR REPLACE INTO llm_requests ({}) VALUES ({})'.format(keys, qs), values)
        conn.commit()
        conn.close()

        super().on_llm_request(data)

    def on_audit_event(self, data: Dict[str, Any]):
        """将审计事件存入数据库"""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        db_data = data.copy()
        keys, qs, values = util.make_insert_auto(db_data)
        cursor.execute('INSERT INTO audit_events ({}) VALUES ({})'.format(keys, qs), values)
        conn.commit()
        conn.close()

        super().on_audit_event(data)
