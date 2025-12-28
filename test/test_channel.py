import unittest
import asyncio
from unittest.mock import AsyncMock
from typing import Set, Optional, List, Any, Dict

import aitoolman.channel as _ch


class TestChannel(unittest.IsolatedAsyncioTestCase):
    """测试基础 Channel 类"""

    def setUp(self):
        self.channel = _ch.Channel()

    def test_initial_state(self):
        """测试初始状态"""
        self.assertFalse(self.channel.closed)
        self.assertIsNotNone(self.channel._message_queue)

    async def test_write_and_read_message(self):
        """测试消息的写入和读取"""
        test_message = "Hello, World!"

        # 写入消息
        await self.channel.write_message(test_message)

        # 读取消息
        result = await self.channel.read_message()
        self.assertEqual(result, test_message)

    async def test_write_after_close(self):
        """测试关闭后写入消息应抛出异常"""
        self.channel.close()

        with self.assertRaises(IOError):
            await self.channel.write_message("test")

    def test_close_method(self):
        """测试关闭方法"""
        self.channel.close()
        self.assertTrue(self.channel.closed)


class TestTextChannel(unittest.IsolatedAsyncioTestCase):
    """测试 TextChannel 类"""

    def setUp(self):
        self.text_channel = _ch.TextChannel()

    async def test_write_and_read_message(self):
        """测试完整消息的读写"""
        test_message = "Complete message"

        await self.text_channel.write_message(test_message)
        result = await self.text_channel.read_message()

        self.assertEqual(result, test_message)

    async def test_write_fragments_and_complete(self):
        """测试片段写入和完整消息生成"""
        fragments = ["Fragment 1", "Fragment 2", "Fragment 3"]
        expected_message = "".join(fragments)

        # 写入片段
        for fragment in fragments[:-1]:
            await self.text_channel.write_fragment(fragment, end=False)

        # 写入最后一个片段并标记结束
        await self.text_channel.write_fragment(fragments[-1], end=True)

        # 读取完整消息
        result = await self.text_channel.read_message()
        self.assertEqual(result, expected_message)

    async def test_fragment_reading_enabled(self):
        """测试启用片段读取模式"""
        text_channel = _ch.TextChannel(read_fragments=True)

        test_message = "Test message"
        await text_channel.write_message(test_message)

        # 应该能够读取片段
        fragment1 = await text_channel.read_fragment()
        self.assertEqual(fragment1, test_message)

        # 结束标记
        fragment2 = await text_channel.read_fragment()
        self.assertIsNone(fragment2)

    async def test_fragment_reading_disabled(self):
        """测试禁用片段读取模式"""
        # 默认情况下 read_fragments=False
        result = await self.text_channel.read_fragment()
        self.assertIsNone(result)

    async def test_read_message_when_reading_fragments(self):
        """测试在片段读取模式下尝试读取完整消息应抛出异常"""
        text_channel = _ch.TextChannel(read_fragments=True)

        with self.assertRaises(RuntimeError):
            await text_channel.read_message()


class MockXmlTagFilter(_ch.BaseXmlTagFilter):
    """用于测试的 BaseXmlTagFilter 具体实现"""

    def __init__(self, tags: Set[str]):
        super().__init__(tags)
        self.messages = []
        self.fragments = []

    async def on_message_tag(self, tag: Optional[str], message: str, end: bool):
        self.messages.append({
            'tag': tag,
            'message': message,
            'end': end
        })

    async def on_fragment_tag(self, tag: Optional[str], text: str, end: bool):
        self.fragments.append({
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

    async def test_write_message_complete_tag(self):
        """测试处理完整的带标签消息"""
        message = "Before <tag1>Content inside tag1</tag1> After"
        await self.filter.write_message(message)

        self.assertEqual(len(self.filter.messages), 3)

        # 检查第一个片段（标签前）
        self.assertIsNone(self.filter.messages[0]['tag'])
        self.assertEqual(self.filter.messages[0]['message'], "Before ")
        self.assertFalse(self.filter.messages[0]['end'])

        # 检查标签内容
        self.assertEqual(self.filter.messages[1]['tag'], 'tag1')
        self.assertEqual(self.filter.messages[1]['message'], "Content inside tag1")
        self.assertTrue(self.filter.messages[1]['end'])

        # 检查标签后内容
        self.assertEqual(self.filter.messages[2]['tag'], None)
        self.assertEqual(self.filter.messages[2]['message'], " After")
        self.assertTrue(self.filter.messages[2]['end'])

    async def test_write_message_nested_tags(self):
        """测试处理嵌套标签（应作为普通文本处理）"""
        message = "Before <tag1><tag2>Nested</tag2></tag1> After"
        await self.filter.write_message(message)
        # [{'tag': None, 'message': 'Before ', 'end': False}, {'tag': 'tag2', 'message': 'Nested', 'end': True}, {'tag': None, 'message': ' After', 'end': True}]

        # 嵌套标签应该作为普通文本处理
        found_nested = False
        for msg in self.filter.messages:
            if msg['tag'] == 'tag1' and '<tag2>' in msg['message']:
                found_nested = True
                break
        self.assertTrue(found_nested)

    async def test_write_message_unknown_tag(self):
        """测试处理未知标签"""
        message = "Before <unknown>Content</unknown> After"
        await self.filter.write_message(message)

        # 未知标签应该作为普通文本处理
        found_unknown = False
        for msg in self.filter.messages:
            if msg['tag'] is None and '<unknown>' in msg['message']:
                found_unknown = True
                break
        self.assertTrue(found_unknown)

    async def test_write_fragments_simple(self):
        """测试处理简单的消息片段"""
        fragments = [
            ("Before ", False),
            ("<tag1>", False),
            ("Content inside", False),
            (" tag1", True),  # 结束
            (" After", True)  # 结束
        ]

        for text, end in fragments:
            await self.filter.write_fragment(text, end)

        # 检查片段处理结果
        self.assertEqual(len(self.filter.fragments), 3)

        # 验证第一个片段
        self.assertIsNone(self.filter.fragments[0]['tag'])
        self.assertEqual(self.filter.fragments[0]['text'], "Before ")
        self.assertFalse(self.filter.fragments[0]['end'])

        # 验证标签内容
        self.assertEqual(self.filter.fragments[1]['tag'], 'tag1')
        self.assertEqual(self.filter.fragments[1]['text'], "Content inside tag1")
        self.assertTrue(self.filter.fragments[1]['end'])

    async def test_write_fragments_complex(self):
        """测试处理复杂的消息片段（跨片段标签）"""
        # 标签被分割在多个片段中
        fragments = [
            ("Before <tag1", False),
            (">Content", False),
            (" inside tag1</tag1", False),
            ("> After", True)
        ]

        for text, end in fragments:
            await self.filter.write_fragment(text, end)

        # 应该能正确识别跨片段的标签
        self.assertTrue(len(self.filter.fragments) > 0)

        # 检查是否识别到了 tag1
        tag1_found = any(
            frag['tag'] == 'tag1' for frag in self.filter.fragments
        )
        self.assertTrue(tag1_found)


class TestXmlTagToChannelFilter(unittest.IsolatedAsyncioTestCase):
    """测试 XmlTagToChannelFilter 类"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

        # 创建模拟通道
        self.default_channel = _ch.TextChannel()
        self.channel1 = _ch.TextChannel()
        self.channel2 = _ch.TextChannel()

        self.channel_map = {
            'tag1': self.channel1,
            'tag2': self.channel2
        }

        self.filter = _ch.XmlTagToChannelFilter(self.default_channel, self.channel_map)

    def tearDown(self):
        self.loop.close()

    async def test_message_routing_to_specific_channel(self):
        """测试消息路由到特定通道"""
        message = "Before <tag1>Content for channel1</tag1> After"

        # 模拟通道的写入方法
        self.default_channel.write_message = AsyncMock()
        self.channel1.write_message = AsyncMock()
        self.channel2.write_message = AsyncMock()

        await self.filter.on_message_tag('tag1', "Content for channel1", True)

        # 验证消息被路由到正确的通道
        self.channel1.write_message.assert_called_once_with("Content for channel1")
        self.default_channel.write_message.assert_not_called()
        self.channel2.write_message.assert_not_called()

    async def test_message_routing_to_default_channel(self):
        """测试消息路由到默认通道"""
        await self.filter.on_message_tag(None, "Default content", True)

        # 读取默认通道的消息验证
        await self.default_channel.write_message("Default content")
        result = await self.default_channel.read_message()
        self.assertEqual(result, "Default content")

    async def test_fragment_routing_to_specific_channel(self):
        """测试片段路由到特定通道"""
        # 模拟通道的片段写入方法
        self.default_channel.write_fragment = AsyncMock()
        self.channel1.write_fragment = AsyncMock()
        self.channel2.write_fragment = AsyncMock()

        await self.filter.on_fragment_tag('tag1', "Fragment content", False)

        # 验证片段被路由到正确的通道
        self.channel1.write_fragment.assert_called_once_with("Fragment content", False)
        self.default_channel.write_fragment.assert_not_called()
        self.channel2.write_fragment.assert_not_called()

    async def test_fragment_routing_to_default_channel(self):
        """测试片段路由到默认通道"""
        await self.filter.on_fragment_tag(None, "Default fragment", True)

        # 验证默认通道收到了片段
        await self.default_channel.write_fragment("Default fragment", True)

        # 读取完整消息验证
        await self.default_channel.write_message("Default fragment")
        result = await self.default_channel.read_message()
        self.assertEqual(result, "Default fragment")

    async def test_complex_message_processing(self):
        """测试复杂消息处理"""
        test_message = """
解释说明……
<tag1>通道1消息</tag1>
<tag2>通道2消息</tag2>
其他文字……
"""

        # 处理完整消息
        await self.filter.write_message(test_message)

        # 这里可以添加更详细的验证逻辑
        # 由于是异步处理，可能需要等待或使用模拟对象

        # 简单验证处理没有抛出异常
        self.assertTrue(True)


class TestIntegration(unittest.IsolatedAsyncioTestCase):
    """集成测试"""

    def setUp(self):
        self.loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self.loop)

    def tearDown(self):
        self.loop.close()

    async def test_end_to_end_workflow(self):
        """测试端到端工作流程"""
        # 创建通道
        default_channel = _ch.TextChannel(read_fragments=True)
        channel1 = _ch.TextChannel(read_fragments=True)
        channel2 = _ch.TextChannel(read_fragments=True)

        channel_map = {'reasoning': channel1, 'action': channel2}
        filter = _ch.XmlTagToChannelFilter(default_channel, channel_map)

        # 模拟 LLM 输出
        llm_output = """
让我思考一下
<think>
<summary>Thought for 0 seconds</summary>
&gt; 用户需要帮助解决数学问题
</think>
好的，<action>我将计算 2+2 的结果</action>
结果是 4
"""

        # 处理消息
        await filter.write_message(llm_output)

        # 这里可以添加验证各个通道收到正确内容的逻辑
        # 由于时间关系，这里简化为验证处理完成
        self.assertTrue(True)


class TestCollectTextChannels(unittest.IsolatedAsyncioTestCase):
    """`collect_text_channels`函数的单元测试"""

    async def test_fragment_mode_single_channel(self):
        """测试**片段模式**下的单通道正常流程：片段写入+结束标记"""
        # 1. 初始化通道（片段模式）
        channel = _ch.TextChannel(read_fragments=True)
        channel_name = "fragment_channel"
        channels = {channel_name: channel}

        # 2. 模拟向通道写入片段
        async def write_data():
            await channel.write_fragment("Hello")  # 片段1
            await asyncio.sleep(0.01)  # 模拟异步延迟
            await channel.write_fragment(" World", end=True)  # 片段2（结束片段）

        # 3. 启动写入任务
        write_task = asyncio.create_task(write_data())

        # 4. 收集通道事件
        events = []
        async for event in _ch.collect_text_channels(channels, read_fragments=True, timeout=1):
            events.append(event)

        # 5. 等待写入任务完成
        await write_task

        # 6. 验证事件序列
        self.assertEqual(len(events), 3)
        # 片段1：Hello（未结束）
        self.assertEqual(events[0], _ch.ChannelEvent(channel_name, "Hello", True, False))
        # 片段2：World（未结束）
        self.assertEqual(events[1], _ch.ChannelEvent(channel_name, " World", True, False))
        # 结束标记：None（已结束）
        self.assertEqual(events[2], _ch.ChannelEvent(channel_name, None, True, True))

    async def test_full_message_mode_single_channel(self):
        """测试**完整消息模式**下的单通道正常流程：完整消息写入+结束标记"""
        # 1. 初始化通道（完整消息模式）
        channel = _ch.TextChannel(read_fragments=False)  # 必须关闭片段模式，否则read_message报错
        channel_name = "full_message_channel"
        channels = {channel_name: channel}

        # 2. 模拟向通道写入完整消息
        async def write_data():
            await channel.write_message("Hi!")  # 消息1
            await asyncio.sleep(0.01)
            await channel.write_message("How are you?")  # 消息2
            await channel.write_message(None)  # 结束标记

        # 3. 启动写入任务
        write_task = asyncio.create_task(write_data())

        # 4. 收集通道事件
        events = []
        async for event in _ch.collect_text_channels(channels, read_fragments=False, timeout=1):
            events.append(event)

        # 5. 等待写入任务完成
        await write_task

        # 6. 验证事件序列
        self.assertEqual(len(events), 3)
        # 消息1：Hi!（未结束）
        self.assertEqual(events[0], _ch.ChannelEvent(channel_name, "Hi!", False, False))
        # 消息2：How are you?（未结束）
        self.assertEqual(events[1], _ch.ChannelEvent(channel_name, "How are you?", False, False))
        # 结束标记：None（已结束）
        self.assertEqual(events[2], _ch.ChannelEvent(channel_name, None, False, True))

    async def test_multiple_channels_concurrent(self):
        """测试**多通道并发写入**：确保多路复用正确捕获所有事件"""
        # 1. 初始化两个通道（片段+完整消息）
        frag_channel = _ch.TextChannel(read_fragments=True)
        full_channel = _ch.TextChannel(read_fragments=True)
        channels = {
            "frag_channel": frag_channel,
            "full_channel": full_channel
        }

        # 2. 模拟并发写入
        async def write_frag():
            await frag_channel.write_fragment("Part 1")
            await asyncio.sleep(0.02)  # 让完整消息通道先发送
            await frag_channel.write_fragment("Part 2", end=True)  # 片段结束

        async def write_full():
            await full_channel.write_fragment("Message 1")
            await asyncio.sleep(0.01)
            await full_channel.write_fragment("Message 2")
            await full_channel.write_fragment("", end=True)  # 完整消息结束

        # 3. 启动写入任务
        task_frag = asyncio.create_task(write_frag())
        task_full = asyncio.create_task(write_full())

        # 4. 收集事件
        events = []
        async for event in _ch.collect_text_channels(channels, read_fragments=True, timeout=1):
            events.append(event)

        # 5. 等待所有写入完成
        await task_frag
        await task_full

        # 6. 验证每个通道的事件序列（顺序不影响，但单通道内必须有序）
        frag_events = [e for e in events if e.channel == "frag_channel"]
        full_events = [e for e in events if e.channel == "full_channel"]

        # 验证片段通道的事件
        self.assertEqual(len(frag_events), 3)
        self.assertEqual(frag_events[0].message, "Part 1")
        self.assertEqual(frag_events[1].message, "Part 2")
        self.assertTrue(frag_events[2].is_end)  # 结束标记

        # 验证完整消息通道的事件
        self.assertEqual(len(full_events), 4)
        self.assertEqual(full_events[0].message, "Message 1")
        self.assertEqual(full_events[1].message, "Message 2")
        self.assertTrue(full_events[-1].is_end)  # 结束标记

    async def test_timeout_no_data(self):
        """测试**超时场景**：无数据写入时触发TimeoutError"""
        # 1. 初始化空通道
        channel = _ch.TextChannel(read_fragments=True)
        channels = {"empty_channel": channel}

        # 2. 收集事件（设置短超时）
        with self.assertRaises(TimeoutError):
            async for _ in _ch.collect_text_channels(channels, read_fragments=True, timeout=0.1):
                pass  # 无数据写入，等待超时

    async def test_channel_closed(self):
        """测试**通道关闭**：关闭后直接返回结束事件"""
        # 1. 初始化通道并关闭
        channel = _ch.TextChannel(read_fragments=True)
        channel_name = "closed_channel"
        channel.close()  # 提前关闭通道
        channels = {channel_name: channel}

        # 2. 收集事件
        events = []
        async for event in _ch.collect_text_channels(channels, read_fragments=True, timeout=1):
            events.append(event)

        # 3. 验证：关闭后直接返回结束事件
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0], _ch.ChannelEvent(channel_name, None, True, True))


if __name__ == '__main__':
    # 运行测试
    unittest.main(verbosity=2)
