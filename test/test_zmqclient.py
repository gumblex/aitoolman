import asyncio
import unittest

from aitoolman import util
from aitoolman.channel import ChannelEvent
from aitoolman.zmqclient import LLMZmqClient


class _RecordingChannel:
    """记录写入事件的输出通道（mock output_channel）"""

    def __init__(self):
        self.events = []

    async def write(self, item, timeout=None):
        self.events.append(item)

    async def write_complete(self, timeout=None):
        pass


class _FakeRequest:
    """仅包含 channel_write 解码所需属性的请求对象"""

    def __init__(self, request_id):
        self.request_id = request_id
        self.output_channel = _RecordingChannel()
        self.is_cancelled = False


class _OneShotSocket:
    """按序返回预设消息，耗尽后抛出 CancelledError 结束监听循环"""

    def __init__(self, messages):
        self.messages = list(messages)

    async def recv_multipart(self):
        if not self.messages:
            raise asyncio.CancelledError()
        return self.messages.pop(0)


class TestZmqClientChannelWriteDecode(unittest.IsolatedAsyncioTestCase):
    """测试客户端对紧凑数组 channel_write 格式的解码"""

    def _make_client(self):
        client = object.__new__(LLMZmqClient)
        client.connected = True
        client.active_requests = {}
        return client

    async def _listen_once(self, client, payload):
        """向客户端喂入一条 channel_write 消息并驱动 listen_responses 处理"""
        client.socket = _OneShotSocket([
            [b'', b'channel_write', util.encode_message(payload)]
        ])
        await client.listen_responses()

    async def test_decode_order_topic_and_none(self):
        """按条目与 token 顺序解码，topic 正确，None 原样透传"""
        client = self._make_client()
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
        client = self._make_client()
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
        client = self._make_client()
        req1 = _FakeRequest('req-1')
        client.active_requests = {'req-1': req1}

        payload = [['req-1', 'response', []]]
        await self._listen_once(client, payload)

        self.assertEqual(req1.output_channel.events, [])


if __name__ == '__main__':
    unittest.main()
