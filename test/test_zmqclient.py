import asyncio
import unittest

from aitoolman import util, zmqprotocol
from aitoolman.channel import ChannelEvent
from aitoolman.model import LLMApiRequestError, Message, ToolCall
from aitoolman.zmqclient import LLMZmqClient
from aitoolman.zmqprotocol import (
    ResponsePayload, RpcResponsePayload, ErrorResponsePayload, CancelAckPayload,
)


class _RecordingChannel:
    """记录写入事件的输出通道（mock output_channel）"""

    def __init__(self):
        self.events = []
        self.write_complete_called = False

    async def write(self, item, timeout=None):
        self.events.append(item)

    async def write_complete(self, timeout=None):
        self.write_complete_called = True


class _FakeRequest:
    """包含 channel_write/handle_response/handle_cancel_ack 所需属性的请求对象"""

    def __init__(self, request_id):
        self.request_id = request_id
        self.output_channel = _RecordingChannel()
        self.is_cancelled = False
        self.response = asyncio.Future()


class _OneShotSocket:
    """按序返回预设消息，耗尽后抛出 CancelledError 结束监听循环"""

    def __init__(self, messages):
        self.messages = list(messages)

    async def recv_multipart(self):
        if not self.messages:
            raise asyncio.CancelledError()
        return self.messages.pop(0)


class _RecordingSocket:
    """记录 send_multipart 帧的假 socket（发送侧测试用）"""

    def __init__(self):
        self.sent_frames = []

    async def send_multipart(self, frames):
        self.sent_frames.append(list(frames))


def _make_client():
    """构造绕过 __init__ 的客户端实例，配置测试所需属性"""
    client = object.__new__(LLMZmqClient)
    client.client_id = 'client-1'
    client.auth_token = b'secret-token'
    client.connected = True
    client.active_requests = {}
    client._rpc_futures = {}
    client.zmq_timeout = 0.5
    return client


def _make_response_data(**overrides):
    """构造服务端 send_response 组装的完整 response dict"""
    data = {
        'client_id': 'client-1',
        'context_id': 'ctx-1',
        'request_id': 'req-1',
        'model_name': 'model-a',
        'stream': False,
        'start_time': 1000.0,
        'queue_time': 0.1,
        'queue_length': 0,
        'time_to_first_token': 0.2,
        'total_response_time': 1.5,
        'response_text': 'hi',
        'response_reasoning': '',
        'response_tool_calls': [],
        'response_meta': {},
        'finish_reason': 'stop',
        'error_text': None,
        'prompt_tokens': 3,
        'completion_tokens': 2,
        'response_message': None,
    }
    data.update(overrides)
    return data


class TestZmqClientChannelWriteDecode(unittest.IsolatedAsyncioTestCase):
    """测试客户端对紧凑数组 channel_write 格式的解码"""

    async def _listen_once(self, client, payload):
        """向客户端喂入一条 channel_write 消息并驱动 listen_responses 处理"""
        client.socket = _OneShotSocket([
            [b'', b'channel_write', util.encode_message(payload)]
        ])
        await client.listen_responses()

    async def test_decode_order_topic_and_none(self):
        """按条目与 token 顺序解码，topic 正确，None 原样透传"""
        client = _make_client()
        req1 = _FakeRequest('req-1')
        req2 = _FakeRequest('req-2')
        client.active_requests = {'req-1': req1, 'req-2': req2}

        payload = [
            ['req-1', 'response', ['a', None, 'b']],
            ['req-2', 'reasoning', [None]],
            ['req-1', 'reasoning', ['思考']],
        ]
        await self._listen_once(client, payload)

        self.assertEqual(req1.output_channel.events, [
            ChannelEvent(topic='response', data='a'),
            ChannelEvent(topic='response', data=None),
            ChannelEvent(topic='response', data='b'),
            ChannelEvent(topic='reasoning', data='思考'),
        ])
        self.assertEqual(req2.output_channel.events, [
            ChannelEvent(topic='reasoning', data=None),
        ])

    async def test_unknown_request_skipped(self):
        """未知 request_id 的条目被跳过，不影响其他条目"""
        client = _make_client()
        req1 = _FakeRequest('req-1')
        client.active_requests = {'req-1': req1}

        payload = [
            ['unknown-req', 'response', ['x']],
            ['req-1', 'response', ['y']],
        ]
        await self._listen_once(client, payload)

        self.assertEqual(req1.output_channel.events, [
            ChannelEvent(topic='response', data='y'),
        ])

    async def test_empty_texts_produces_no_events(self):
        """空 texts 列表不产生任何事件"""
        client = _make_client()
        req1 = _FakeRequest('req-1')
        client.active_requests = {'req-1': req1}

        payload = [['req-1', 'response', []]]
        await self._listen_once(client, payload)

        self.assertEqual(req1.output_channel.events, [])


class TestZmqClientSendMessage(unittest.IsolatedAsyncioTestCase):
    """测试发送侧消息：四帧结构与 zmqprotocol Struct 载荷字段正确"""

    def _check_frame(self, socket, index, msg_type):
        """校验第 index 条消息的前三帧格式，返回载荷 bytes"""
        frames = socket.sent_frames[index]
        self.assertEqual(frames[0], b'')  # DEALER 空前缀帧
        self.assertEqual(frames[1], b'secret-token')  # auth token
        self.assertEqual(frames[2], msg_type.encode('utf-8'))  # msg_type
        return frames[3]

    async def test_send_message_frame_layout(self):
        """_send_message 发送 [b'', auth_token, msg_type, payload] 四帧"""
        client = _make_client()
        client.socket = _RecordingSocket()

        await client._send_message(
            'rpc', zmqprotocol.RpcPayload(request_id='rpc-1', method='list_models', kwargs={}))

        payload_bytes = self._check_frame(client.socket, 0, 'rpc')
        payload = zmqprotocol.decode_client_payload('rpc', payload_bytes)
        self.assertEqual(payload.request_id, 'rpc-1')
        self.assertEqual(payload.method, 'list_models')

    async def test_request_sends_request_payload(self):
        """request 发送 RequestPayload，登记 active_requests，字段与请求一致"""
        client = _make_client()
        client.socket = _RecordingSocket()
        messages = [Message(role='user', content='hello')]
        tools = {'get_weather': {'type': 'function', 'function': {'name': 'get_weather'}}}

        request = await client.request(
            'model-a', messages, tools=tools, options={'temperature': 0.5},
            stream=True, context_id='ctx-1')

        self.assertIs(client.active_requests[request.request_id], request)

        payload_bytes = self._check_frame(client.socket, 0, 'request')
        payload = zmqprotocol.decode_client_payload('request', payload_bytes)
        self.assertEqual(payload.client_id, 'client-1')
        self.assertEqual(payload.context_id, 'ctx-1')
        self.assertEqual(payload.request_id, request.request_id)
        self.assertEqual(payload.model_name, 'model-a')
        self.assertEqual(payload.messages, [m.to_dict() for m in messages])
        self.assertEqual(payload.messages[0]['role'], 'user')
        self.assertEqual(payload.messages[0]['content'], 'hello')
        self.assertEqual(payload.tools, tools)
        self.assertEqual(payload.options, {'temperature': 0.5})
        self.assertTrue(payload.stream)

    async def test_request_not_connected_raises(self):
        """未连接时 request 抛 RuntimeError，不发送消息"""
        client = _make_client()
        client.connected = False
        client.socket = _RecordingSocket()

        with self.assertRaises(RuntimeError):
            await client.request('model-a', [Message(role='user', content='hi')])
        self.assertEqual(client.socket.sent_frames, [])

    async def test_cancel_sends_cancel_payload(self):
        """cancel 发送 CancelPayload 并触发 output_channel.write_complete"""
        client = _make_client()
        client.socket = _RecordingSocket()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        await client.cancel('req-1')

        payload_bytes = self._check_frame(client.socket, 0, 'cancel')
        payload = zmqprotocol.decode_client_payload('cancel', payload_bytes)
        self.assertEqual(payload.client_id, 'client-1')
        self.assertEqual(payload.request_id, 'req-1')
        self.assertTrue(req.output_channel.write_complete_called)

    async def test_cancel_unknown_request_no_message(self):
        """cancel 未知 request_id 仅告警，不发送消息"""
        client = _make_client()
        client.socket = _RecordingSocket()

        await client.cancel('missing')

        self.assertEqual(client.socket.sent_frames, [])

    async def test_cancel_all_sends_cancel_all_payload(self):
        """cancel_all 发送 CancelAllPayload 并对所有活跃请求 write_complete"""
        client = _make_client()
        client.socket = _RecordingSocket()
        req1 = _FakeRequest('req-1')
        req2 = _FakeRequest('req-2')
        client.active_requests = {'req-1': req1, 'req-2': req2}

        await client.cancel_all(context_id='ctx-1')

        payload_bytes = self._check_frame(client.socket, 0, 'cancel_all')
        payload = zmqprotocol.decode_client_payload('cancel_all', payload_bytes)
        self.assertEqual(payload.client_id, 'client-1')
        self.assertEqual(payload.context_id, 'ctx-1')
        self.assertTrue(req1.output_channel.write_complete_called)
        self.assertTrue(req2.output_channel.write_complete_called)

    async def test_audit_event_sends_audit_payload(self):
        """audit_event 发送 AuditEventPayload，kwargs 进入 data 字段"""
        client = _make_client()
        client.socket = _RecordingSocket()

        await client.audit_event('ctx-1', 'custom_event', key='value', count=3)

        payload_bytes = self._check_frame(client.socket, 0, 'audit_event')
        payload = zmqprotocol.decode_client_payload('audit_event', payload_bytes)
        self.assertEqual(payload.client_id, 'client-1')
        self.assertEqual(payload.context_id, 'ctx-1')
        self.assertEqual(payload.event_type, 'custom_event')
        self.assertEqual(payload.data, {'key': 'value', 'count': 3})
        self.assertIsNotNone(payload.timestamp)

    async def test_send_rpc_roundtrip(self):
        """_send_rpc 发送 RpcPayload 并被 rpc_response 唤醒返回结果"""
        client = _make_client()
        socket = _RecordingSocket()
        client.socket = socket

        async def _feed_response():
            for _ in range(1000):
                if socket.sent_frames:
                    break
                await asyncio.sleep(0.001)
            sent = zmqprotocol.decode_client_payload('rpc', socket.sent_frames[0][3])
            client.handle_rpc_response(
                sent.request_id,
                RpcResponsePayload(request_id=sent.request_id, result=['model-a']))

        result, _ = await asyncio.gather(
            client._send_rpc('list_models', {'tag': 'chat'}),
            _feed_response())
        self.assertEqual(result, ['model-a'])

    async def test_send_rpc_timeout(self):
        """无响应时 _send_rpc 超时抛 TimeoutError 并清理 future"""
        client = _make_client()
        client.socket = _RecordingSocket()
        client.zmq_timeout = 0.01

        with self.assertRaises(TimeoutError):
            await client._send_rpc('list_models', {'tag': 'chat'})

        self.assertEqual(client._rpc_futures, {})


class TestZmqClientReceiveDecode(unittest.IsolatedAsyncioTestCase):
    """测试接收侧各 msg_type 的 Struct 解码与处理逻辑"""

    async def _listen_once(self, client, msg_type, payload):
        """喂入一条指定 msg_type 的 Struct 载荷并驱动 listen_responses"""
        client.socket = _OneShotSocket([
            [b'fake-server', msg_type.encode('utf-8'), zmqprotocol.encode_payload(payload)]
        ])
        await client.listen_responses()

    async def test_response_with_meta(self):
        """response 载荷构造 LLMProviderResponse，response_meta 原样保留"""
        client = _make_client()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        response_data = _make_response_data(
            response_meta={'usage': {'prompt_tokens': 3}},
            response_message={'role': 'assistant', 'content': 'hi'})
        await self._listen_once(client, 'response',
                                ResponsePayload(request_id='req-1', response=response_data))

        self.assertNotIn('req-1', client.active_requests)
        response = req.response.result()
        self.assertEqual(response.client_id, 'client-1')
        self.assertEqual(response.context_id, 'ctx-1')
        self.assertEqual(response.request_id, 'req-1')
        self.assertEqual(response.model_name, 'model-a')
        self.assertFalse(response.stream)
        self.assertEqual(response.response_text, 'hi')
        self.assertEqual(response.finish_reason, 'stop')
        self.assertEqual(response.prompt_tokens, 3)
        self.assertEqual(response.completion_tokens, 2)
        self.assertEqual(response.response_meta, {'usage': {'prompt_tokens': 3}})
        self.assertEqual(response.response_message.role, 'assistant')
        self.assertEqual(response.response_message.content, 'hi')
        # 有正文时不写 EOF 事件
        self.assertEqual(req.output_channel.events, [])

    async def test_response_without_meta_falls_back_to_empty_dict(self):
        """旧版对端缺失 response_meta 字段时回退为空 dict"""
        client = _make_client()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        response_data = _make_response_data()
        del response_data['response_meta']
        await self._listen_once(client, 'response',
                                ResponsePayload(request_id='req-1', response=response_data))

        response = req.response.result()
        self.assertEqual(response.response_meta, {})

    async def test_response_empty_content_writes_eof_events(self):
        """response_text/reasoning 均为空时先写 reasoning/response 的 None EOF 事件"""
        client = _make_client()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        response_data = _make_response_data(response_text='', response_reasoning='')
        await self._listen_once(client, 'response',
                                ResponsePayload(request_id='req-1', response=response_data))

        self.assertEqual(req.output_channel.events, [
            ChannelEvent('reasoning', None),
            ChannelEvent('response', None),
        ])

    async def test_response_with_tool_calls(self):
        """response_tool_calls dict 列表转换为 ToolCall 对象"""
        client = _make_client()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        response_data = _make_response_data(
            response_text='',
            response_tool_calls=[{
                'name': 'get_weather',
                'arguments_text': '{"city": "北京"}',
                'arguments': {'city': '北京'},
                'id': 'tc-1',
                'type': 'function',
            }],
            finish_reason='tool_calls')
        await self._listen_once(client, 'response',
                                ResponsePayload(request_id='req-1', response=response_data))

        response = req.response.result()
        self.assertEqual(response.finish_reason, 'tool_calls')
        self.assertEqual(len(response.response_tool_calls), 1)
        tool = response.response_tool_calls[0]
        self.assertIsInstance(tool, ToolCall)
        self.assertEqual(tool.name, 'get_weather')
        self.assertEqual(tool.arguments, {'city': '北京'})
        self.assertEqual(tool.id, 'tc-1')

    async def test_response_for_unknown_request_skipped(self):
        """未知 request_id 的 response 仅告警跳过"""
        client = _make_client()
        await self._listen_once(client, 'response',
                                ResponsePayload(request_id='nope', response={}))

    async def test_rpc_response_sets_result(self):
        """rpc_response 解码后 set_result 到对应 future"""
        client = _make_client()
        future = asyncio.Future()
        client._rpc_futures['rpc-1'] = future

        await self._listen_once(client, 'rpc_response',
                                RpcResponsePayload(request_id='rpc-1', result={'models': ['m1']}))

        self.assertTrue(future.done())
        self.assertEqual(future.result(), {'models': ['m1']})

    async def test_rpc_response_known_error_type(self):
        """已知 error_type 实例化为 model 模块中的异常类"""
        client = _make_client()
        future = asyncio.Future()
        client._rpc_futures['rpc-1'] = future

        await self._listen_once(client, 'rpc_response',
                                RpcResponsePayload(request_id='rpc-1',
                                                   error_type='LLMApiRequestError',
                                                   error_text='LLMApiRequestError: boom'))

        self.assertTrue(future.done())
        with self.assertRaises(LLMApiRequestError):
            future.result()

    async def test_rpc_response_unknown_error_type(self):
        """未知 error_type 回退为 RuntimeError"""
        client = _make_client()
        future = asyncio.Future()
        client._rpc_futures['rpc-1'] = future

        await self._listen_once(client, 'rpc_response',
                                RpcResponsePayload(request_id='rpc-1',
                                                   error_type='SomeCustomError',
                                                   error_text='SomeCustomError: boom'))

        self.assertTrue(future.done())
        with self.assertRaises(RuntimeError):
            future.result()

    async def test_cancel_ack_marks_cancelled(self):
        """cancel_ack 置 is_cancelled 并移除活跃请求"""
        client = _make_client()
        req = _FakeRequest('req-1')
        client.active_requests['req-1'] = req

        await self._listen_once(client, 'cancel_ack', CancelAckPayload(request_id='req-1'))

        self.assertTrue(req.is_cancelled)
        self.assertNotIn('req-1', client.active_requests)

    async def test_error_payload_does_not_crash(self):
        """error 载荷记录日志，监听循环正常退出"""
        client = _make_client()
        await self._listen_once(client, 'error',
                                ErrorResponsePayload(type='error', request_id='req-1', error='boom'))

    async def test_unknown_msg_type_skipped(self):
        """未知 msg_type 记录 warning 并跳过"""
        client = _make_client()
        client.socket = _OneShotSocket([[b'fake-server', b'unknown_type', b'{}']])
        await client.listen_responses()

    async def test_invalid_frame_length_skipped(self):
        """帧数不为 3 时记录 error 并跳过"""
        client = _make_client()
        client.socket = _OneShotSocket([[b'fake-server', b'response']])
        await client.listen_responses()


if __name__ == '__main__':
    unittest.main()
