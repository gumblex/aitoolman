
import unittest
import asyncio
import sys
from io import StringIO
from unittest.mock import patch
from typing import Set, Optional, List, Any, Dict

import aitoolman.channel as _ch


class TestChannel(unittest.IsolatedAsyncioTestCase):
    """测试基础 Channel 类"""

    def setUp(self):
        self.channel = _ch.Channel()

    async def test_write_and_read(self):
        """测试消息的写入和读取"""
        test_message = _ch.ChannelEvent(topic="test", data="Hello, World!")

        # 写入消息
        await self.channel.write(test_message)

        # 读取消息
        result = await self.channel.read()
        self.assertEqual(result, test_message)

    async def test_eof_error(self):
        """测试读取EOF时抛出异常"""
        await self.channel.write_complete()

        with self.assertRaises(EOFError):
            await self.channel.read()

    async def test_multiple_writes_and_reads(self):
        """测试多次写入和读取"""
        messages = [
            _ch.ChannelEvent(topic="test", data="Message 1"),
            _ch.ChannelEvent(topic="test", data="Message 2"),
            _ch.ChannelEvent(topic="test", data="Message 3")
        ]

        for msg in messages:
            await self.channel.write(msg)

        for expected in messages:
            result = await self.channel.read()
            self.assertEqual(result, expected)


class TestTopicWriterReader(unittest.IsolatedAsyncioTestCase):
    """测试 TopicWriter 和 TopicReader 类"""

    async def test_write_and_read_specific_topic(self):
        """测试指定topic的读写"""
        writer = _ch.TopicWriter({'reasoning', 'response'})
        reasoning_reader = writer.reader('reasoning')
        response_reader = writer.reader('response')

        # 写入不同topic内容
        await writer.write(_ch.ChannelEvent('reasoning', '我现在要计算2+2'))
        await writer.write(_ch.ChannelEvent('response', '2+2的结果是4'))
        await writer.write(_ch.ChannelEvent('reasoning', None))
        await writer.write(_ch.ChannelEvent('response', None))
        await writer.write_complete()

        # 读取reasoning topic内容
        event1 = await reasoning_reader.read()
        self.assertEqual(event1.topic, 'reasoning')
        self.assertEqual(event1.data, '我现在要计算2+2')
        event2 = await reasoning_reader.read()
        self.assertEqual(event2.data, None)
        with self.assertRaises(EOFError):
            await reasoning_reader.read()

        # 读取response topic内容
        event1 = await response_reader.read()
        self.assertEqual(event1.topic, 'response')
        self.assertEqual(event1.data, '2+2的结果是4')
        await response_reader.read()
        with self.assertRaises(EOFError):
            await response_reader.read()

    async def test_unknown_topic_write_error(self):
        """测试写入未知topic抛出异常"""
        writer = _ch.TopicWriter({'reasoning', 'response'})
        with self.assertRaises(ValueError):
            await writer.write(_ch.ChannelEvent('unknown', 'test'))


class MockXmlTagFilter(_ch.BaseXmlTagFilter):
    """用于测试的 BaseXmlTagFilter 具体实现"""

    def __init__(self, tags: Set[str]):
        super().__init__(tags)
        self.tag_calls: List[Dict[str, Any]] = []

    async def on_tag(self, tag: Optional[str], text: str, end: bool):
        self.tag_calls.append({
            'tag': tag,
            'text': text,
            'end': end
        })


class TestBaseXmlTagFilter(unittest.IsolatedAsyncioTestCase):
    """测试 BaseXmlTagFilter 类"""

    def setUp(self):
        self.tags = {'tag1', 'tag2', 'tag3'}
        self.filter = MockXmlTagFilter(self.tags)

    def test_initialization(self):
        """测试初始化"""
        self.assertEqual(self.filter.tags, self.tags)
        self.assertIsNone(self.filter.current_tag)
        self.assertEqual(self.filter.current_content, [])

    async def test_write_complete_message(self):
        """测试处理完整消息"""
        message = "Before <tag1>Content inside tag1</tag1> After"
        await self.filter.write(_ch.ChannelEvent(topic='response', data=message))
        await self.filter.write(_ch.ChannelEvent(topic='response', data=None))  # 结束消息

        # 应该有三个调用：Before, tag1内容, After
        self.assertEqual(len(self.filter.tag_calls), 4)

        # 检查第一个调用（Before）
        self.assertIsNone(self.filter.tag_calls[0]['tag'])
        self.assertEqual(self.filter.tag_calls[0]['text'], "Before ")
        self.assertFalse(self.filter.tag_calls[0]['end'])

        # 检查tag1内容
        self.assertEqual(self.filter.tag_calls[1]['tag'], 'tag1')
        self.assertEqual(self.filter.tag_calls[1]['text'], "Content inside tag1")
        self.assertTrue(self.filter.tag_calls[1]['end'])

        # 检查After
        self.assertIsNone(self.filter.tag_calls[2]['tag'])
        self.assertEqual(self.filter.tag_calls[2]['text'], " After")
        self.assertTrue(self.filter.tag_calls[-1]['end'])

    async def test_write_fragments(self):
        """测试处理消息片段"""
        fragments = [
            "Before <tag1",
            ">Content",
            " inside",
            " tag1</tag1",
            "> After"
        ]

        for fragment in fragments:
            await self.filter.write(_ch.ChannelEvent(topic='response', data=fragment))

        await self.filter.write(_ch.ChannelEvent(topic='response', data=None))  # 结束消息

        # 应该能正确识别跨片段的标签
        tag1_calls = [call for call in self.filter.tag_calls if call['tag'] == 'tag1']
        self.assertEqual(len(tag1_calls), 1)
        self.assertEqual(tag1_calls[0]['text'], "Content inside tag1")
        self.assertTrue(tag1_calls[0]['end'])

    async def test_nested_tags(self):
        """测试嵌套标签"""
        message = "Before <tag1><tag2>Nested</tag2></tag1> After"
        await self.filter.write(_ch.ChannelEvent(topic='response', data=message))
        await self.filter.write(_ch.ChannelEvent(topic='response', data=None))

        # tag1应该包含嵌套标签作为普通文本
        tag1_calls = [call for call in self.filter.tag_calls if call['tag'] == 'tag1']
        self.assertEqual(len(tag1_calls), 1)
        self.assertIn('<tag2>', tag1_calls[0]['text'])

    async def test_unknown_tag(self):
        """测试未知标签"""
        message = "Before <unknown>Content</unknown> After"
        await self.filter.write(_ch.ChannelEvent(topic='response', data=message))
        await self.filter.write(_ch.ChannelEvent(topic='response', data=None))

        # 所有内容应该作为普通文本处理
        none_tag_calls = [call for call in self.filter.tag_calls if call['tag'] is None]
        self.assertTrue(len(none_tag_calls) > 0)

        # 检查是否有包含未知标签的文本
        texts = ' '.join(call['text'] for call in none_tag_calls)
        self.assertIn('<unknown>', texts)
        self.assertIn('Content', texts)


class TestXmlTagToChannelFilter(unittest.IsolatedAsyncioTestCase):
    """测试 XmlTagToChannelFilter 类"""

    def setUp(self):
        self.output_channel = _ch.Channel()
        self.filter = _ch.XmlTagToChannelFilter(
            output_channel=self.output_channel,
            tags={'tag1', 'tag2'}
        )

    async def test_routing_to_specific_topic(self):
        """测试路由到特定topic"""
        # 写入tag1内容
        await self.filter.on_tag('tag1', "Content for tag1", True)

        # 应该能读到tag1的内容
        event1 = await self.output_channel.read()
        self.assertEqual(event1.topic, 'tag1')
        self.assertEqual(event1.data, "Content for tag1")

        # 应该收到None标记结束
        event2 = await self.output_channel.read()
        self.assertEqual(event2.topic, 'tag1')
        self.assertIsNone(event2.data)

    async def test_routing_to_none_topic(self):
        """测试普通文本路由到None topic"""
        await self.filter.on_tag(None, "Default content", True)

        event1 = await self.output_channel.read()
        self.assertIsNone(event1.topic)
        self.assertEqual(event1.data, "Default content")

        event2 = await self.output_channel.read()
        self.assertIsNone(event2.topic)
        self.assertIsNone(event2.data)

    async def test_complex_message_processing(self):
        """测试复杂消息处理"""
        test_message = """
解释说明……
<tag1>通道1消息</tag1>
<tag2>通道2消息</tag2>
其他文字……
"""

        # 处理完整消息
        await self.filter.write(_ch.ChannelEvent(topic='response', data=test_message))
        await self.filter.write(_ch.ChannelEvent(topic='response', data=None))  # 结束
        await self.filter.write_complete()

        # 收集所有事件
        events = []
        while True:
            try:
                events.append(await self.output_channel.read())
            except EOFError:
                break

        # 验证tag1内容
        tag1_events = [e for e in events if e.topic == 'tag1' and e.data is not None]
        self.assertEqual(len(tag1_events), 1)
        self.assertEqual(tag1_events[0].data, "通道1消息")

        # 验证tag2内容
        tag2_events = [e for e in events if e.topic == 'tag2' and e.data is not None]
        self.assertEqual(len(tag2_events), 1)
        self.assertEqual(tag2_events[0].data, "通道2消息")

        # 验证默认内容
        default_content = ''.join([e.data for e in events if e.topic is None and e.data is not None])
        self.assertIn("解释说明", default_content)
        self.assertIn("其他文字", default_content)


class TestPrintChannelOutput(unittest.IsolatedAsyncioTestCase):
    """测试 print_channel_output 工具函数"""

    async def test_output_format(self):
        """测试输出格式正确"""
        channel = _ch.Channel()
        topic_names = {
            'reasoning': '思考过程',
            'response': '最终回复'
        }

        # 写入测试数据
        await channel.write(_ch.ChannelEvent('reasoning', '我要计算2+2'))
        await channel.write(_ch.ChannelEvent('reasoning', None))
        await channel.write(_ch.ChannelEvent('response', '2+2=4'))
        await channel.write(_ch.ChannelEvent('response', None))
        await channel.write_complete()

        # 捕获输出
        captured_output = StringIO()
        with patch('sys.stdout', new=captured_output):
            await _ch.print_channel_output(channel, topic_names, header=True)

        output = captured_output.getvalue()
        self.assertIn('===== 思考过程 =====', output)
        self.assertIn('我要计算2+2', output)
        self.assertIn('===== 最终回复 =====', output)
        self.assertIn('2+2=4', output)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """集成测试"""

    async def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 创建输出通道
        output_channel = _ch.Channel()

        # 创建XML标签过滤器
        filter = _ch.XmlTagToChannelFilter(
            output_channel=output_channel,
            tags={'reasoning', 'action'}
        )

        # 模拟 LLM 输出
        llm_output = """
让我思考一下
<reasoning>我需要计算2+2</reasoning>
<action>计算 2+2</action>
结果是4
"""

        # 处理消息
        await filter.write(_ch.ChannelEvent(topic='response', data=llm_output))
        await filter.write(_ch.ChannelEvent(topic='response', data=None))  # 结束
        await output_channel.write_complete()

        # 收集所有事件
        events = []
        while True:
            try:
                events.append(await output_channel.read())
            except EOFError:
                break

        # 验证各个topic的内容
        reasoning_content = ''.join([e.data for e in events if e.topic == 'reasoning' and e.data is not None])
        self.assertEqual(reasoning_content, "我需要计算2+2")

        action_content = ''.join([e.data for e in events if e.topic == 'action' and e.data is not None])
        self.assertEqual(action_content, "计算 2+2")

        default_content = ''.join([e.data for e in events if e.topic is None and e.data is not None])
        self.assertIn("让我思考一下", default_content)
        self.assertIn("结果是4", default_content)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
