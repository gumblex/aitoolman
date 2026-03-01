import json
import secrets
import asyncio
import logging
from typing import Dict, Optional, List, Any

import zmq
import zmq.asyncio

from . import util
from .model import LLMProviderRequest, LLMProviderResponse, FinishReason, Message
from .provider import LLMProviderManager
from .channel import TextFragmentChannel

logger = logging.getLogger(__name__)


class ZmqTextChannel(TextFragmentChannel):
    """适配ZeroMQ的TextChannel，写入时触发Server发送channel_write消息"""
    def __init__(self, server: 'LLMZmqServer', request_id: str, channel_type: str):
        super().__init__()
        self.server = server
        self.request_id = request_id
        self.channel_type = channel_type  # "response" 或 "reasoning"

    async def write(self, message: Optional[str]):
        await self.server.send_channel_write(self.request_id, self.channel_type, "fragment", message)


class LLMZmqServer:
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.ctx = zmq.asyncio.Context()
        self.router_socket = self.ctx.socket(zmq.ROUTER)  # 处理客户端请求
        self.pub_socket = self.ctx.socket(zmq.PUB)        # 发布审计日志
        self.provider_manager = LLMProviderManager(config)
        self.active_requests: Dict[str, LLMProviderRequest] = {}  # request_id -> LLMProviderRequest
        self.auth_token: Optional[str] = config['server'].get('zmq_auth_token')  # 读取认证令牌
        self.authenticated_clients = set()  # 存储已认证的 client_id
        self.running = False

    async def initialize(self):
        """初始化ZeroMQ和ProviderManager"""
        server_config = self.config['server']
        # 绑定ROUTER（处理请求）和PUB（审计日志）
        self.router_socket.bind(server_config['zmq_router_rpc'])
        logger.info(f"Microservice socket (ROUTER) bound to {server_config['zmq_router_rpc']}")
        if server_config.get('zmq_pub_event'):
            socket_type = server_config.get('zmq_pub_event_type') or 'bind'
            if socket_type == 'connect':
                self.pub_socket.connect(server_config['zmq_pub_event'])
            elif socket_type == 'bind':  # bind
                self.pub_socket.bind(server_config['zmq_pub_event'])
            elif not socket_type:
                pass
            else:
                raise ValueError("Invalid zmq_pub_event_type: %s" %  socket_type)
            logger.info(f"Audit socket (PUB) bound to {server_config['zmq_pub_event']}")
        else:
            logger.info(f"Audit socket (PUB) disabled")
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
        if self.auth_token:
            if msg_type == 'auth':
                await self.handle_auth(client_id, json_data)
                return
            elif client_id not in self.authenticated_clients:
                logger.warning("[%s] Unauthenticated client", client_id)
                if msg_type == 'request':
                    await self.handle_request_auth_failed(client_id, json_data)
                else:
                    await self.send_error(client_id, request_id, "Authentication required")
                return

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
        output_channel = ZmqTextChannel(self, request_id, 'response')
        reasoning_channel = ZmqTextChannel(self, request_id, 'reasoning')

        # 初始化LLMRequest
        request = LLMProviderRequest(
            client_id=client_id,
            context_id=context_id,
            request_id=request_id,
            model_name=model_name,
            messages=messages,
            tools=tools,
            options=options,
            stream=stream,
            output_channel=output_channel,
            reasoning_channel=reasoning_channel
        )
        self.active_requests[request_id] = request
        logger.info("[%s] Start request. model: %s, stream: %s", request_id, model_name, stream)
        self.provider_manager.process_request(request, self.on_request_completed)

    async def handle_request_auth_failed(self, client_id: str, json_data: Dict[str, Any]):
        request_id = json_data.get('request_id') or util.get_id()
        model_name = json_data['model_name']
        stream = json_data.get('stream', False)
        context_id = json_data.get('context_id')

        response = LLMProviderResponse(
            client_id=client_id,
            context_id=context_id or "",
            request_id=request_id,
            model_name=model_name,
            stream=stream,
            finish_reason=FinishReason.error_request.value,
            error_text='ZeroMQ authentication failed'
        )
        await self.send_response(client_id, request_id, response)

    async def on_request_completed(self, request: LLMProviderRequest):
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

    async def send_channel_write(self, request_id: str, channel_type: str, mode: str, text: str):
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
            'text': text
        }
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def send_response(self, client_id: str, request_id: str, response: LLMProviderResponse):
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
                'response_tool_calls': [tc._asdict() for tc in response.response_tool_calls],
                'finish_reason': response.finish_reason.value if isinstance(response.finish_reason, FinishReason) else response.finish_reason,
                'error_text': response.error_text,
                'prompt_tokens': response.prompt_tokens,
                'completion_tokens': response.completion_tokens,
                'response_message': response.response_message.to_dict() if response.response_message else None
            }
        }
        # logger.debug("send_msg: %s", message)
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def handle_auth(self, client_id: str, json_data: Dict[str, Any]):
        """处理认证消息"""
        token = json_data.get('token')
        request_id = json_data.get('request_id')

        if secrets.compare_digest(token, self.auth_token):
            self.authenticated_clients.add(client_id)
            await self.send_auth_response(client_id, request_id, success=True)
            logger.info("[%s] Client authenticated successfully", client_id)
        else:
            await self.send_auth_response(client_id, request_id, success=False)
            logger.warning("[%s] Client authentication failed", client_id)

    async def send_auth_response(self, client_id: str, request_id: str, success: bool):
        """发送认证响应"""
        message = {
            'type': 'auth_response',
            'request_id': request_id,
            'success': success
        }
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def send_error(self, client_id: str, request_id: str, error: str):
        """发送错误消息"""
        message = {
            'type': 'error',
            'request_id': request_id,
            'error': error
        }
        await self.router_socket.send_multipart([
            client_id.encode('utf-8'),
            b'',
            util.encode_message(message)
        ])

    async def publish_audit_log(self, request: LLMProviderRequest):
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
                [tc._asdict() for tc in response.response_tool_calls]
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

