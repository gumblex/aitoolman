import json
import time
import asyncio
import logging
import dataclasses
from typing import Dict, Optional, List, Any

import zmq
import zmq.asyncio

from . import util
from .model import LLMRequest, LLMResponse, FinishReason, Message
from .provider import LLMProviderManager
from .channel import TextChannel

logger = logging.getLogger(__name__)


class ZmqTextChannel(TextChannel):
    """适配ZeroMQ的TextChannel，写入时触发Server发送channel_write消息"""
    def __init__(self, server: 'LLMZmqServer', request_id: str, channel_type: str):
        super().__init__(read_fragments=True)
        self.server = server
        self.request_id = request_id
        self.channel_type = channel_type  # "response" 或 "reasoning"

    async def write_fragment(self, text: str, end: bool = False):
        await self.server.send_channel_write(self.request_id, self.channel_type, "fragment", text, end)

    async def write_message(self, text: str):
        await self.server.send_channel_write(self.request_id, self.channel_type, "message", text, end=False)


class LLMZmqServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ctx = zmq.asyncio.Context()
        self.router_socket = self.ctx.socket(zmq.ROUTER)  # 处理客户端请求
        self.pub_socket = self.ctx.socket(zmq.PUB)        # 发布审计日志
        self.provider_manager = LLMProviderManager(config)
        self.active_requests: Dict[str, LLMRequest] = {}  # request_id -> LLMRequest
        self.running = False

    async def initialize(self):
        """初始化ZeroMQ和ProviderManager"""
        server_config = self.config['server']
        # 绑定ROUTER（处理请求）和PUB（审计日志）
        self.router_socket.bind(server_config['zmq_router_rpc'])
        self.pub_socket.bind(server_config['zmq_pub_event'])
        logger.info(f"ROUTER bound to {server_config['zmq_router_rpc']}")
        logger.info(f"PUB bound to {server_config['zmq_pub_event']}")
        # 初始化ProviderManager
        await self.provider_manager.initialize()

    async def run(self):
        """启动服务主循环"""
        self.running = True
        await self.initialize()
        try:
            while self.running:
                message = await self.router_socket.recv_multipart()
                asyncio.create_task(self.process_message(message))
        except (asyncio.CancelledError, KeyboardInterrupt):
            logger.info("Server shutting down...")
        finally:
            await self.cleanup()

    async def process_message(self, message: List[bytes]):
        """解析并处理客户端消息"""
        if len(message) != 3:
            logger.error(f"Invalid message format: {len(message)} parts")
            return
        client_id = message[0].decode('utf-8')
        json_data = json.loads(message[2].decode('utf-8'))
        msg_type = json_data.get('type')
        request_id = json_data.get('request_id')

        logger.debug("[%s] Request: %s", client_id, json_data)
        if msg_type == 'request':
            await self.handle_request(client_id, json_data)
        elif msg_type == 'cancel':
            await self.handle_cancel(client_id, request_id)
        elif msg_type == 'cancel_all':
            await self.handle_cancel_all(client_id, json_data.get('context_id'))
        elif msg_type == 'audit_event':
            await self.handle_audit_event(client_id, json_data)
        else:
            logger.warning(f"Unknown message type: {msg_type}")

    async def handle_request(self, client_id: str, json_data: Dict[str, Any]):
        """处理客户端请求"""
        # 构造LLMRequest
        request_id = json_data.get('request_id') or util.get_id()
        model_name = json_data['model_name']
        messages = [Message.from_dict(m) for m in json_data['messages']]
        tools = json_data.get('tools') or {}
        options = json_data.get('options') or {}
        stream = json_data.get('stream', False)
        context_id = json_data.get('context_id')

        # 创建ZmqTextChannel（捕获channel写入并推送）
        response_channel = ZmqTextChannel(self, request_id, 'response')
        reasoning_channel = ZmqTextChannel(self, request_id, 'reasoning')

        # 初始化LLMRequest
        request = LLMRequest(
            client_id=client_id,
            context_id=context_id,
            request_id=request_id,
            model_name=model_name,
            messages=messages,
            tools=tools,
            options=options,
            stream=stream,
            response_channel=response_channel,
            reasoning_channel=reasoning_channel
        )
        self.active_requests[request_id] = request
        logger.info("[%s] Start request. model: %s, stream: %s", request_id, model_name, stream)
        self.provider_manager.process_request(request, self.on_request_completed)

    async def on_request_completed(self, request: LLMRequest):
        """请求完成后的回调（发送结果+审计）"""
        response = request.response.result()
        if not response:
            logger.warning("[%s] Request has no response", request.request_id)
            return

        client_id = request.client_id
        # 发送结果（error或response）
        await self.send_response(client_id, request.request_id, response)

        # 发布审计日志
        await self.publish_audit_log(request)
        # 清理活跃请求
        del self.active_requests[request.request_id]

    async def send_channel_write(self, request_id: str, channel_type: str, mode: str, text: str, end: bool):
        """发送channel写入消息给客户端"""
        request = self.active_requests.get(request_id)
        if not request:
            logger.warning("[%s] Request not found for channel write", request_id)
            return
        client_id = request.client_id
        message = {
            'type': 'channel_write',
            'request_id': request_id,
            'channel': channel_type,
            'mode': mode,
            'text': text,
            'end': end
        }
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def send_response(self, client_id: str, request_id: str, response: LLMResponse):
        """发送完整响应消息"""
        message = {
            'type': 'response',
            'request_id': request_id,
            'response': {
                'client_id': response.client_id,
                'context_id': response.context_id,
                'request_id': response.request_id,
                'model_name': response.model_name,
                'stream': response.stream,
                'start_time': response.start_time,
                'queue_time': response.queue_time,
                'queue_length': response.queue_length,
                'time_to_first_token': response.time_to_first_token,
                'total_response_time': response.total_response_time,
                'response_text': response.response_text,
                'response_reasoning': response.response_reasoning,
                'response_tool_calls': [dataclasses.asdict(tc) for tc in response.response_tool_calls],
                'finish_reason': response.finish_reason.value if isinstance(response.finish_reason, FinishReason) else response.finish_reason,
                'error_text': response.error_text,
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens,
                'response_message': response.response_message
            }
        }
        # logger.debug("send_msg: %s", message)
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def publish_audit_log(self, request: LLMRequest):
        """发布审计日志到PUB socket"""
        response = request.response.result()
        audit_log = {
            'client_id': response.client_id,
            'context_id': response.context_id,
            'request_id': response.request_id,
            'model_name': response.model_name,
            'stream': response.stream,
            'start_time': response.start_time,
            'queue_time': response.queue_time,
            'queue_length': response.queue_length,
            'time_to_first_token': response.time_to_first_token,
            'total_response_time': response.total_response_time,
            'response_text': response.response_text,
            'response_reasoning': response.response_reasoning,
            'response_tool_calls': (
                [dataclasses.asdict(tc) for tc in response.response_tool_calls]
                if response.response_tool_calls else None),
            'finish_reason': response.finish_reason,
            'error_text': response.error_text,
            'prompt_tokens': response.prompt_tokens,
            'completion_tokens': response.completion_tokens,
            'request_messages': [m.to_dict() for m in request.messages],
            'request_tools': request.tools,
            'request_options': request.options,
            'response_message': response.response_message
        }
        await self.pub_socket.send_multipart([
            b'llm_request',
            util.encode_message(audit_log)
        ])

    async def handle_cancel(self, client_id: str, request_id: str):
        """处理取消请求"""
        request = self.active_requests.get(request_id)
        if not request or request.client_id != client_id:
            logger.warning(f"Invalid cancel request for {request_id}")
            return
        await self.provider_manager.cancel_request(request_id)
        await self.send_cancel_ack(client_id, request_id)

    async def handle_cancel_all(self, client_id: str, context_id: Optional[str]):
        """处理取消所有请求"""
        await self.provider_manager.cancel_all_requests(client_id, context_id)
        logger.info(f"Cancelled all requests for client {client_id} (context {context_id})")

    async def send_cancel_ack(self, client_id: str, request_id: str):
        """发送取消确认"""
        message = {
            'type': 'cancel_ack',
            'request_id': request_id
        }
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def handle_audit_event(self, client_id: str, json_data: Dict[str, Any]):
        """处理审计事件消息并发布到PUB接口"""
        event_data = {
            'client_id': client_id,
            'context_id': json_data.get('context_id'),
            'event_type': json_data.get('event_type'),
            'data': json_data.get('data', {}),
            'timestamp': json_data.get('timestamp')
        }
        await self.pub_socket.send_multipart([
            b'audit_event',
            util.encode_message(event_data)
        ])

    async def cleanup(self):
        """清理资源"""
        await self.provider_manager.cleanup()
        self.router_socket.close()
        self.pub_socket.close()
        self.ctx.term()
        logger.info("Server cleaned up")

