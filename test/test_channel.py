
import unittest
import asyncio
from unittest.mock import AsyncMock, patch
from typing import Set, Optional, List, Any, Dict

import aitoolman.channel as _ch


class TestChannel(unittest.IsolatedAsyncioTestCase):
    """测试基础 Channel 类"""

    def setUp(self):
        self.channel = _ch.Channel()

    async def test_write_and_read(self):
        """测试消息的写入和读取"""
        test_message = "Hello, World!"

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
        messages = ["Message 1", "Message 2", "Message 3"]

        for msg in messages:
            await self.channel.write(msg)

        for expected in messages:
            result = await self.channel.read()
            self.assertEqual(result, expected)


class TestTextFragmentChannel(unittest.IsolatedAsyncioTestCase):
    """测试 TextFragmentChannel 类"""

    def setUp(self):
        self.channel = _ch.TextFragmentChannel()

    async def test_write_and_read_fragments(self):
        """测试片段的写入和读取"""
        fragments = ["Hello", " ", "World", "!"]

        for fragment in fragments:
            await self.channel.write(fragment)

        # 结束消息
        await self.channel.write(None)

        # 读取所有片段
        result = await self.channel.read_whole_message()
        self.assertEqual(result, "Hello World!")

    async def test_multiple_messages(self):
        """测试多个消息的读写"""
        # 第一个消息
        await self.channel.write("First")
        await self.channel.write("Message")
        await self.channel.write(None)

        # 第二个消息
        await self.channel.write("Second")
        await self.channel.write("Message")
        await self.channel.write(None)

        # 读取第一个消息
        result1 = await self.channel.read_whole_message()
        self.assertEqual(result1, "FirstMessage")

        # 读取第二个消息
        result2 = await self.channel.read_whole_message()
        self.assertEqual(result2, "SecondMessage")

    async def test_eof_behavior(self):
        """测试EOF行为"""
        await self.channel.write("Before EOF")
        await self.channel.write_complete()  # 发送EOF

        # 应该能正常读取到EOF前的数据
        result = await self.channel.read()
        self.assertEqual(result, "Before EOF")

        # 下一次读取应该抛出EOFError
        with self.assertRaises(EOFError):
            await self.channel.read()

    async def test_read_whole_message_with_eof(self):
        """测试在EOF后调用read_whole_message"""
        await self.channel.write("Test")
        await self.channel.write(None)  # 结束消息
        await self.channel.write_complete()  # 结束流

        # 应该能读取完整消息
        result = await self.channel.read_whole_message()
        self.assertEqual(result, "Test")

        # 再次调用应该抛出EOFError
        with self.assertRaises(EOFError):
            await self.channel.read_whole_message()


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
        await self.filter.write(message)
        await self.filter.write(None)  # 结束消息

        # 应该有三个调用：Before, tag1内容, After
        self.assertEqual(len(self.filter.tag_calls), 3)

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
        self.assertTrue(self.filter.tag_calls[2]['end'])

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
            await self.filter.write(fragment)

        await self.filter.write(None)  # 结束消息

        # 应该能正确识别跨片段的标签
        tag1_calls = [call for call in self.filter.tag_calls if call['tag'] == 'tag1']
        self.assertEqual(len(tag1_calls), 1)
        self.assertEqual(tag1_calls[0]['text'], "Content inside tag1")
        self.assertTrue(tag1_calls[0]['end'])

    async def test_nested_tags(self):
        """测试嵌套标签"""
        message = "Before <tag1><tag2>Nested</tag2></tag1> After"
        await self.filter.write(message)
        await self.filter.write(None)

        # tag2应该被识别
        tag2_calls = [call for call in self.filter.tag_calls if call['tag'] == 'tag2']
        self.assertEqual(len(tag2_calls), 1)
        self.assertEqual(tag2_calls[0]['text'], "Nested")

        # tag1应该包含嵌套标签作为普通文本
        tag1_calls = [call for call in self.filter.tag_calls if call['tag'] == 'tag1']
        self.assertEqual(len(tag1_calls), 1)
        self.assertIn('<tag2>', tag1_calls[0]['text'])

    async def test_unknown_tag(self):
        """测试未知标签"""
        message = "Before <unknown>Content</unknown> After"
        await self.filter.write(message)
        await self.filter.write(None)

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
        # 创建通道
        self.default_channel = _ch.TextFragmentChannel()
        self.channel1 = _ch.TextFragmentChannel()
        self.channel2 = _ch.TextFragmentChannel()

        self.channel_map = {
            'tag1': self.channel1,
            'tag2': self.channel2
        }

        self.filter = _ch.XmlTagToChannelFilter(self.default_channel, self.channel_map)

    async def test_routing_to_specific_channel(self):
        """测试路由到特定通道"""
        # 写入tag1内容
        await self.filter.on_tag('tag1', "Content for channel1", True)

        # 应该能在channel1中读取到
        result = await self.channel1.read()
        self.assertEqual(result, "Content for channel1")

        # 应该收到None标记
        eof_result = await self.channel1.read()
        self.assertIsNone(eof_result)

    async def test_routing_to_default_channel(self):
        """测试路由到默认通道"""
        await self.filter.on_tag(None, "Default content", True)

        result = await self.default_channel.read()
        self.assertEqual(result, "Default content")

        eof_result = await self.default_channel.read()
        self.assertIsNone(eof_result)

    async def test_complex_message_processing(self):
        """测试复杂消息处理"""
        test_message = """
解释说明……
<tag1>通道1消息</tag1>
<tag2>通道2消息</tag2>
其他文字……
"""

        # 处理完整消息
        await self.filter.write(test_message)
        await self.filter.write(None)  # 结束

        # 验证tag1通道
        tag1_content = await self.channel1.read_whole_message()
        self.assertEqual(tag1_content, "通道1消息")

        # 验证tag2通道
        tag2_content = await self.channel2.read_whole_message()
        self.assertEqual(tag2_content, "通道2消息")

        # 验证默认通道包含其他内容
        default_content = await self.default_channel.read_whole_message()
        self.assertIn("解释说明", default_content)
        self.assertIn("其他文字", default_content)


class MockChannelCollector(_ch.ChannelCollector):
    """用于测试的 ChannelCollector 实现"""

    def __init__(self, channels: Dict[str, _ch.Channel]):
        super().__init__(channels)
        self.events: List[Dict[str, Any]] = []

    async def on_channel_start(self, channel_name: str):
        self.events.append({'type': 'start', 'channel': channel_name})

    async def on_channel_read(self, channel_name: str, message):
        self.events.append({
            'type': 'read',
            'channel': channel_name,
            'message': message
        })

    async def on_channel_end(self, channel_name: str):
        self.events.append({'type': 'end', 'channel': channel_name})

    async def on_channel_eof(self, channel_name: str):
        self.events.append({'type': 'eof', 'channel': channel_name})


class TestChannelCollector(unittest.IsolatedAsyncioTestCase):
    """测试 ChannelCollector 类"""

    def setUp(self):
        self.channel1 = _ch.TextFragmentChannel()
        self.channel2 = _ch.TextFragmentChannel()
        self.channels = {
            "channel1": self.channel1,
            "channel2": self.channel2
        }

    async def test_single_channel_single_message(self):
        """测试单通道单消息"""
        collector = MockChannelCollector({"channel1": self.channel1})

        # 写入数据
        await self.channel1.write("Hello")
        await self.channel1.write(None)  # 结束消息
        await self.channel1.write_complete()  # 结束流

        # 启动收集器（在后台运行）
        listen_task = asyncio.create_task(collector.start_listening())

        # 等待处理完成
        await asyncio.sleep(0.1)

        # 停止收集器
        collector.close()

        # 等待任务完成
        await listen_task

        # 验证事件
        start_events = [e for e in collector.events if e['type'] == 'start']
        read_events = [e for e in collector.events if e['type'] == 'read']
        end_events = [e for e in collector.events if e['type'] == 'end']
        eof_events = [e for e in collector.events if e['type'] == 'eof']

        self.assertEqual(len(start_events), 1)
        self.assertEqual(len(read_events), 1)
        self.assertEqual(read_events[0]['message'], "Hello")
        self.assertEqual(len(end_events), 1)
        self.assertEqual(len(eof_events), 1)

    async def test_multiple_channels_concurrent(self):
        """测试多通道并发"""
        collector = MockChannelCollector(self.channels)

        # 向两个通道写入数据
        await self.channel1.write("Message 1 from channel1")
        await self.channel1.write(None)

        await self.channel2.write("Message 1 from channel2")
        await self.channel2.write(None)

        await self.channel1.write_complete()
        await self.channel2.write_complete()

        # 启动收集器
        listen_task = asyncio.create_task(collector.start_listening())

        # 等待处理
        await asyncio.sleep(0.1)

        # 停止收集器
        collector.close()

        # 等待任务完成
        await listen_task

        # 验证两个通道都收到了数据
        channel1_reads = [e for e in collector.events
                         if e['type'] == 'read' and e['channel'] == 'channel1']
        channel2_reads = [e for e in collector.events
                         if e['type'] == 'read' and e['channel'] == 'channel2']

        self.assertEqual(len(channel1_reads), 1)
        self.assertEqual(len(channel2_reads), 1)
        self.assertIn("channel1", channel1_reads[0]['message'])
        self.assertIn("channel2", channel2_reads[0]['message'])


class TestDefaultTextChannelCollector(unittest.IsolatedAsyncioTestCase):
    """测试 DefaultTextChannelCollector 类"""

    def setUp(self):
        self.channel1 = _ch.TextFragmentChannel()
        self.channel2 = _ch.TextFragmentChannel()
        self.channels = {
            "reasoning": self.channel1,
            "response": self.channel2
        }

    async def test_output_format(self):
        """测试输出格式"""
        collector = _ch.DefaultTextChannelCollector(self.channels)

        # 捕获打印输出
        with patch('builtins.print') as mock_print:
            # 写入数据
            await self.channel1.write("Thinking...")
            await self.channel1.write(None)
            await self.channel1.write_complete()

            await self.channel2.write("Final answer")
            await self.channel2.write(None)
            await self.channel2.write_complete()

            # 启动收集器（在后台运行）
            listen_task = asyncio.create_task(collector.start_listening())

            # 等待处理
            await asyncio.sleep(0.1)

            # 停止收集器
            collector.close()

            # 等待任务完成
            await listen_task

            # 验证打印调用
            print_calls = [str(call) for call in mock_print.call_args_list]
            self.assertTrue(len(print_calls) > 0)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """集成测试"""

    async def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 创建通道
        default_channel = _ch.TextFragmentChannel()
        reasoning_channel = _ch.TextFragmentChannel()
        action_channel = _ch.TextFragmentChannel()

        channel_map = {
            'reasoning': reasoning_channel,
            'action': action_channel
        }

        filter = _ch.XmlTagToChannelFilter(default_channel, channel_map)

        # 模拟 LLM 输出
        llm_output = """
让我思考一下
<reasoning>我需要计算2+2</reasoning>
<action>计算 2+2</action>
结果是4
"""

        # 处理消息
        await filter.write(llm_output)
        await filter.write(None)  # 结束

        # 验证各个通道的内容
        reasoning_content = await reasoning_channel.read_whole_message()
        self.assertEqual(reasoning_content, "我需要计算2+2")

        action_content = await action_channel.read_whole_message()
        self.assertEqual(action_content, "计算 2+2")

        default_content = await default_channel.read_whole_message()
        self.assertIn("让我思考一下", default_content)
        self.assertIn("结果是4", default_content)


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
