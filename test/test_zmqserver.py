import asyncio
import unittest
from typing import List

import msgspec

from aitoolman import util, zmqprotocol
from aitoolman.zmqserver import LLMZmqServer, ZmqServerChannel
from aitoolman.zmqprotocol import ZmqChannelMessageBuffer, ResponsePayload, CancelAckPayload
from aitoolman.channel import ChannelEvent
from aitoolman.model import LLMProviderRequest, LLMProviderResponse


def _make_server(auth_token=None, manage_token=None):
    """创建测试用服务器实例，可设置两种token"""
    server = object.__new__(LLMZmqServer)
    server.zmq_auth_token = auth_token
    server.zmq_manage_token = manage_token
    return server


class TestCheckPermission(unittest.TestCase):
    """测试权限校验逻辑"""

    def test_manage_permission_none_manage_token(self):
        """管理token为None时，任何人都没有管理权限"""
        server = _make_server(manage_token=None)
        # 任何token都不应有管理权限
        self.assertFalse(server._check_permission('any_token', manage=True))
        self.assertFalse(server._check_permission('', manage=True))
        self.assertFalse(server._check_permission('valid_token', manage=True))

    def test_manage_permission_valid_token(self):
        """正确管理token通过校验"""
        server = _make_server(manage_token='manage_secret')
        self.assertTrue(server._check_permission('manage_secret', manage=True))

    def test_manage_permission_invalid_token(self):
        """错误管理token被拒绝"""
        server = _make_server(manage_token='manage_secret')
        self.assertFalse(server._check_permission('wrong_token', manage=True))

    def test_manage_permission_empty_token_match(self):
        """空管理token匹配空字符串"""
        server = _make_server(manage_token='')
        self.assertTrue(server._check_permission('', manage=True))
        self.assertFalse(server._check_permission('some_token', manage=True))

    def test_normal_permission_no_auth_token(self):
        """普通token为None时，任何token都有普通权限"""
        server = _make_server(auth_token=None)
        # 即使管理token为None，只要auth_token为None就有普通权限
        self.assertTrue(server._check_permission('any_token', manage=False))
        self.assertTrue(server._check_permission('', manage=False))

        # 设置管理token后，普通权限仍然可用
        server_with_manage = _make_server(auth_token=None, manage_token='manage_secret')
        self.assertTrue(server_with_manage._check_permission('any_token', manage=False))
        self.assertTrue(server_with_manage._check_permission('', manage=False))

    def test_normal_permission_with_auth_token(self):
        """普通token不为None时，只有匹配的token有普通权限"""
        server = _make_server(auth_token='auth_secret')
        self.assertTrue(server._check_permission('auth_secret', manage=False))
        self.assertFalse(server._check_permission('wrong_token', manage=False))
        self.assertFalse(server._check_permission('', manage=False))

    def test_normal_permission_with_both_tokens(self):
        """同时设置auth和manage token时，两种token都有普通权限"""
        server = _make_server(auth_token='auth_secret', manage_token='manage_secret')
        # auth_token有普通权限
        self.assertTrue(server._check_permission('auth_secret', manage=False))
        # manage_token也有普通权限
        self.assertTrue(server._check_permission('manage_secret', manage=False))
        # 其他token没有普通权限
        self.assertFalse(server._check_permission('other_token', manage=False))

    def test_normal_permission_empty_auth_token(self):
        """普通token为空字符串时，只有空token有普通权限"""
        server = _make_server(auth_token='')
        self.assertTrue(server._check_permission('', manage=False))
        self.assertFalse(server._check_permission('any_token', manage=False))

        # 同时设置manage_token
        server_with_manage = _make_server(auth_token='', manage_token='manage_secret')
        self.assertTrue(server_with_manage._check_permission('', manage=False))
        self.assertTrue(server_with_manage._check_permission('manage_secret', manage=False))
        self.assertFalse(server_with_manage._check_permission('other_token', manage=False))


def _make_channel_server():
    """创建仅带缓冲相关状态的测试服务器，并用列表收集发送内容"""
    server = object.__new__(LLMZmqServer)
    server.active_requests = {}
    server.channel_msg_buffer = {}
    server.router_lock = asyncio.Lock()
    server.sent_messages = []

    async def _record_send(client_id, msg_type, message):
        server.sent_messages.append((client_id, msg_type, message))

    server._send_client_message = _record_send
    return server


def _make_provider_request(request_id, client_id='client-1'):
    """创建仅用于缓冲测试的最小 LLMProviderRequest"""
    return LLMProviderRequest(
        client_id=client_id,
        context_id='ctx-1',
        request_id=request_id,
        model_name='test-model',
        messages=[],
    )


class TestZmqChannelMessageBuffer(unittest.TestCase):
    """测试 ZmqChannelMessageBuffer 新签名（request_id/topic）与聚合逻辑"""

    def test_new(self):
        """new() 二元签名：request_id/topic/text 正确"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', 'hello')
        self.assertEqual(buffer.request_id, 'req-1')
        self.assertEqual(buffer.topic, 'response')
        self.assertEqual(buffer.text, ['hello'])

    def test_new_with_none_data(self):
        """None（EOF 标记）原样进入缓冲"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', None)
        self.assertEqual(buffer.text, [None])

    def test_add_same_key_merges_in_place(self):
        """同 key（request_id+topic 相同）追加：原地合并并返回 None"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', 'a')
        result = buffer.add('req-1', 'response', 'b')
        self.assertIsNone(result)
        self.assertEqual(buffer.request_id, 'req-1')
        self.assertEqual(buffer.topic, 'response')
        self.assertEqual(buffer.text, ['a', 'b'])

    def test_add_same_key_keeps_none_order(self):
        """同 key 合并时 None 按插入顺序保留"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', 'a')
        buffer.add('req-1', 'response', None)
        buffer.add('req-1', 'response', 'b')
        self.assertEqual(buffer.text, ['a', None, 'b'])

    def test_add_different_request_returns_new_buffer(self):
        """不同 request_id 返回新 buffer，原 buffer 不受影响"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', 'a')
        new_buffer = buffer.add('req-2', 'response', 'b')
        self.assertIsNotNone(new_buffer)
        self.assertEqual(new_buffer.request_id, 'req-2')
        self.assertEqual(new_buffer.topic, 'response')
        self.assertEqual(new_buffer.text, ['b'])
        self.assertEqual(buffer.request_id, 'req-1')
        self.assertEqual(buffer.text, ['a'])

    def test_add_different_topic_returns_new_buffer(self):
        """不同 topic 返回新 buffer，原 buffer 不受影响"""
        buffer = ZmqChannelMessageBuffer.new('req-1', 'response', 'a')
        new_buffer = buffer.add('req-1', 'reasoning', 'b')
        self.assertIsNotNone(new_buffer)
        self.assertEqual(new_buffer.request_id, 'req-1')
        self.assertEqual(new_buffer.topic, 'reasoning')
        self.assertEqual(new_buffer.text, ['b'])
        self.assertEqual(buffer.text, ['a'])


class TestServerChannelBuffer(unittest.IsolatedAsyncioTestCase):
    """测试服务端 send_channel_write 缓冲（Dict 语义）与 _flush_channel_messages 聚合"""

    async def asyncSetUp(self):
        self.server = _make_channel_server()
        self.server.active_requests['req-1'] = _make_provider_request('req-1')

    async def test_send_channel_write_flushes_when_lock_free(self):
        """无发送方占用锁时立即 flush 单条 token（紧凑数组格式）"""
        await self.server.send_channel_write('req-1', ChannelEvent('response', 'a'))
        self.assertEqual(self.server.channel_msg_buffer, {})
        self.assertEqual(self.server.sent_messages, [
            ('client-1', 'channel_write',
             [ZmqChannelMessageBuffer.new('req-1', 'response', 'a')]),
        ])

    async def test_send_channel_write_buffers_when_lock_held(self):
        """锁被占用时仅入 Dict 缓冲，同一流合并为一条条目"""
        await self.server.router_lock.acquire()
        try:
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'a'))
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'b'))
        finally:
            self.server.router_lock.release()

        self.assertEqual(self.server.sent_messages, [])
        buffers = self.server.channel_msg_buffer['client-1']
        self.assertEqual(len(buffers), 1)
        self.assertEqual(buffers[0].request_id, 'req-1')
        self.assertEqual(buffers[0].topic, 'response')
        self.assertEqual(buffers[0].text, ['a', 'b'])

        await self.server._flush_channel_messages()
        self.assertEqual(self.server.sent_messages, [
            ('client-1', 'channel_write',
             [ZmqChannelMessageBuffer('req-1', 'response', ['a', 'b'])]),
        ])
        self.assertEqual(self.server.channel_msg_buffer, {})

    async def test_send_channel_write_new_key_appends_buffer(self):
        """不同 request/topic 在同一 client 缓冲中形成独立条目"""
        self.server.active_requests['req-2'] = _make_provider_request('req-2')
        await self.server.router_lock.acquire()
        try:
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'a'))
            await self.server.send_channel_write('req-1', ChannelEvent('reasoning', 'r'))
            await self.server.send_channel_write('req-2', ChannelEvent('response', 'b'))
        finally:
            self.server.router_lock.release()

        buffers = self.server.channel_msg_buffer['client-1']
        self.assertEqual([(b.request_id, b.topic) for b in buffers], [
            ('req-1', 'response'),
            ('req-1', 'reasoning'),
            ('req-2', 'response'),
        ])

    async def test_send_channel_write_tail_same_key_aggregates(self):
        """仅尾部同 key 聚合：非连续的同 key 流形成独立条目"""
        await self.server.router_lock.acquire()
        try:
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'a'))
            await self.server.send_channel_write('req-1', ChannelEvent('reasoning', 'r'))
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'b'))
        finally:
            self.server.router_lock.release()

        buffers = self.server.channel_msg_buffer['client-1']
        # response(a) 与 response(b) 不相邻，各自独立；reasoning 独立
        self.assertEqual(
            [(b.request_id, b.topic, list(b.text)) for b in buffers], [
                ('req-1', 'response', ['a']),
                ('req-1', 'reasoning', ['r']),
                ('req-1', 'response', ['b']),
            ])

    async def test_send_channel_write_unknown_request_ignored(self):
        """未知 request_id 仅记录告警，不产生缓冲与发送"""
        await self.server.send_channel_write('unknown', ChannelEvent('response', 'a'))
        self.assertEqual(self.server.channel_msg_buffer, {})
        self.assertEqual(self.server.sent_messages, [])

    async def test_flush_aggregates_by_client(self):
        """flush 输出紧凑数组：按 client 分组独立发送、EOF(None) 原样保留、发送后删 key"""
        self.server.active_requests['req-2'] = _make_provider_request('req-2', client_id='client-2')
        await self.server.router_lock.acquire()
        try:
            # 持锁期间仅入缓冲：连续同流 token 合并为 ['a','b']，
            # 不同 topic、EOF(None) 与不同 client 各保持独立条目与顺序
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'a'))
            await self.server.send_channel_write('req-1', ChannelEvent('response', 'b'))
            await self.server.send_channel_write('req-1', ChannelEvent('reasoning', None))
            await self.server.send_channel_write('req-1', ChannelEvent('response', None))
            await self.server.send_channel_write('req-2', ChannelEvent('response', 'x'))
        finally:
            self.server.router_lock.release()
        self.assertEqual(self.server.sent_messages, [])

        await self.server._flush_channel_messages()

        self.assertEqual(self.server.channel_msg_buffer, {})
        self.assertEqual(self.server.sent_messages, [
            ('client-1', 'channel_write', [
                ZmqChannelMessageBuffer('req-1', 'response', ['a', 'b']),
                ZmqChannelMessageBuffer('req-1', 'reasoning', [None]),
                ZmqChannelMessageBuffer('req-1', 'response', [None]),
            ]),
            ('client-2', 'channel_write', [
                ZmqChannelMessageBuffer('req-2', 'response', ['x']),
            ]),
        ])

    async def test_flush_empty_buffer_sends_nothing(self):
        """空缓冲 flush 不发送任何消息"""
        await self.server._flush_channel_messages()
        self.assertEqual(self.server.sent_messages, [])
        self.assertEqual(self.server.channel_msg_buffer, {})

    async def test_flush_drains_messages_added_during_send(self):
        """flush 发送期间新入缓冲的消息由 while 循环继续发送，无遗留"""
        sent = []

        async def _send_with_late_append(client_id, msg_type, message):
            sent.append((client_id, msg_type, message))
            if len(sent) == 1:
                self.server.channel_msg_buffer.setdefault('client-3', []).append(
                    ZmqChannelMessageBuffer.new('req-3', 'response', 'late'))

        self.server._send_client_message = _send_with_late_append
        self.server.channel_msg_buffer = {
            'client-1': [ZmqChannelMessageBuffer.new('req-1', 'response', 'a')],
        }
        await self.server._flush_channel_messages()

        self.assertEqual(len(sent), 2)
        self.assertEqual(sent[0], (
            'client-1', 'channel_write',
            [ZmqChannelMessageBuffer('req-1', 'response', ['a'])]))
        self.assertEqual(sent[1], (
            'client-3', 'channel_write',
            [ZmqChannelMessageBuffer('req-3', 'response', ['late'])]))
        self.assertEqual(self.server.channel_msg_buffer, {})

    async def test_send_response_flushes_buffer_and_carries_meta(self):
        """send_response 先 flush 再发送响应，且 response dict 始终携带 response_meta"""
        self.server.channel_msg_buffer = {
            'client-1': [ZmqChannelMessageBuffer.new('req-1', 'response', 'token')],
        }
        response = LLMProviderResponse(
            client_id='client-1',
            context_id='ctx-1',
            request_id='req-1',
            model_name='test-model',
            stream=True,
        )
        response.response_meta = {'usage': {'prompt_tokens': 3}}
        await self.server.send_response('client-1', 'req-1', response)

        self.assertEqual(len(self.server.sent_messages), 2)
        self.assertEqual(self.server.sent_messages[0], (
            'client-1', 'channel_write',
            [ZmqChannelMessageBuffer('req-1', 'response', ['token'])]))
        self.assertEqual(self.server.sent_messages[1][0], 'client-1')
        self.assertEqual(self.server.sent_messages[1][1], 'response')
        payload = self.server.sent_messages[1][2]
        self.assertIsInstance(payload, ResponsePayload)
        self.assertEqual(payload.request_id, 'req-1')
        self.assertEqual(payload.response['request_id'], 'req-1')
        self.assertEqual(payload.response['response_meta'],
                         {'usage': {'prompt_tokens': 3}})
        self.assertEqual(self.server.channel_msg_buffer, {})

    async def test_send_response_empty_meta_still_carried(self):
        """response_meta 为空 dict 时仍随响应携带"""
        response = LLMProviderResponse(
            client_id='client-1',
            context_id='ctx-1',
            request_id='req-1',
            model_name='test-model',
            stream=False,
        )
        await self.server.send_response('client-1', 'req-1', response)
        payload = self.server.sent_messages[0][2]
        self.assertEqual(payload.response['response_meta'], {})

    async def test_send_cancel_ack_flushes_buffer_first(self):
        """send_cancel_ack 在 router_lock 内先 flush 再发送取消确认"""
        self.server.channel_msg_buffer = {
            'client-1': [ZmqChannelMessageBuffer.new('req-1', 'response', 'token')],
        }
        await self.server.send_cancel_ack('client-1', 'req-1')

        self.assertEqual(len(self.server.sent_messages), 2)
        self.assertEqual(self.server.sent_messages[0], (
            'client-1', 'channel_write',
            [ZmqChannelMessageBuffer('req-1', 'response', ['token'])]))
        self.assertEqual(self.server.sent_messages[1][0], 'client-1')
        self.assertEqual(self.server.sent_messages[1][1], 'cancel_ack')
        self.assertEqual(self.server.sent_messages[1][2],
                         CancelAckPayload(request_id='req-1'))
        self.assertEqual(self.server.channel_msg_buffer, {})


class TestPublishAuditLog(unittest.IsolatedAsyncioTestCase):
    """测试 publish_audit_log 发布审计日志（含 response_meta）"""

    async def test_publish_audit_log_includes_response_meta(self):
        """审计日志携带 response_meta 字段"""
        server = _make_channel_server()
        frames = []

        class _PubSocket:
            async def send_multipart(self, msg):
                frames.append(msg)

        server.pub_socket = _PubSocket()
        request = _make_provider_request('req-1')
        response = LLMProviderResponse(
            client_id='client-1', context_id='ctx-1', request_id='req-1',
            model_name='test-model', stream=True)
        response.response_meta = {'usage': {'prompt_tokens': 3}}
        request.response.set_result(response)

        await server.publish_audit_log(request)

        self.assertEqual(len(frames), 1)
        topic, data = frames[0]
        self.assertEqual(topic, b'llm_request')
        audit = util.decode_json(data)
        self.assertEqual(audit['request_id'], 'req-1')
        self.assertEqual(audit['client_id'], 'client-1')
        self.assertEqual(audit['response_meta'], {'usage': {'prompt_tokens': 3}})


class TestZmqServerChannelWrite(unittest.IsolatedAsyncioTestCase):
    """测试 ZmqServerChannel.write 对 topic/data 的转发"""

    async def test_write_forwards_topic_and_data(self):
        """write 使用 item.topic/item.data 调用 send_channel_write"""
        server = _make_channel_server()
        server.active_requests['req-1'] = _make_provider_request('req-1')
        channel = ZmqServerChannel(server, 'req-1')

        await channel.write(ChannelEvent('response', 'hello'))

        self.assertEqual(server.sent_messages, [
            ('client-1', 'channel_write',
             [ZmqChannelMessageBuffer('req-1', 'response', ['hello'])]),
        ])


class TestZmqProtocolEncodeDecode(unittest.TestCase):
    """zmqprotocol 载荷 Struct 与 ZmqChannelMessageBuffer 编解码往返测试"""

    def test_channel_message_buffer_roundtrip(self):
        """ZmqChannelMessageBuffer 列表编码为紧凑数组并可解码往返"""
        buffers = [ZmqChannelMessageBuffer.new('req-1', 'response', 'a'),
                   ZmqChannelMessageBuffer.new('req-1', 'reasoning', None)]
        data = util.encode_message(buffers)
        # NamedTuple 编码为 [request_id, topic, [text...]] 紧凑数组
        self.assertEqual(util.decode_json(data), [
            ['req-1', 'response', ['a']],
            ['req-1', 'reasoning', [None]],
        ])
        decoded = msgspec.json.decode(
            data, type=List[ZmqChannelMessageBuffer], strict=False)
        self.assertEqual(decoded, buffers)

    def test_client_payload_roundtrip(self):
        """客户端→服务端载荷 Struct 编解码往返"""
        payload = zmqprotocol.RequestPayload(
            client_id='client-1', context_id='ctx-1', request_id='req-1',
            model_name='model-a', stream=True,
            messages=[{'role': 'user', 'content': 'hi'}],
            tools={}, options={'temperature': 0.5})
        decoded = zmqprotocol.decode_client_payload(
            'request', zmqprotocol.encode_payload(payload))
        self.assertEqual(decoded, payload)

    def test_server_payload_roundtrip(self):
        """服务端→客户端载荷 Struct 编解码往返"""
        payload = zmqprotocol.ResponsePayload(
            request_id='req-1', response={'response_meta': {'a': 1}})
        decoded = zmqprotocol.decode_server_payload(
            'response', zmqprotocol.encode_payload(payload))
        self.assertEqual(decoded, payload)

    def test_decode_unknown_msg_type_returns_none(self):
        """未知消息类型解码返回 None"""
        with self.assertRaises(ValueError):
            zmqprotocol.decode_client_payload('no_such_type', b'{}')
        with self.assertRaises(ValueError):
            zmqprotocol.decode_server_payload('no_such_type', b'{}')


if __name__ == '__main__':
    unittest.main()
