import abc
import re
import sys
import asyncio
from typing import Any, Optional, NamedTuple, Protocol, Set, Dict, List


class ChannelEvent(NamedTuple):
    topic: str
    data: Any


class ChannelReader(Protocol):
    async def read(self) -> ChannelEvent:
        ...


class ChannelWriter(Protocol):
    async def write(self, message: ChannelEvent):
        ...

    async def write_complete(self):
        ...


class Channel:
    def __init__(self):
        self._message_queue: asyncio.Queue[Optional[ChannelEvent]] = asyncio.Queue()
        self._eof_received = False

    async def read(self) -> ChannelEvent:
        if self._eof_received:
            raise EOFError("Channel has been closed by peer")

        item = await self._message_queue.get()
        if item is None:
            self._eof_received = True
            raise EOFError("Channel has been closed by peer")
        return item

    async def write(self, message: ChannelEvent):
        await self._message_queue.put(message)

    async def write_complete(self):
        await self._message_queue.put(None)


class TopicWriter(ChannelWriter):
    def __init__(self, topics: Set[str]):
        self._message_queues: Dict[str, asyncio.Queue[Optional[ChannelEvent]]] = {
            t: asyncio.Queue() for t in topics}

    @classmethod
    def default(cls):
        return TopicWriter({'reasoning', 'response'})

    async def write(self, message: ChannelEvent):
        queue = self._message_queues.get(message.topic)
        if queue is None:
            raise ValueError("topic not defined: " + message.topic)
        await queue.put(message)

    async def write_complete(self):
        for queue in self._message_queues.values():
            await queue.put(None)

    def reader(self, topic: str):
        return TopicReader(topic, self._message_queues[topic])


class TopicReader(ChannelReader):
    def __init__(self, topic: str, queue: asyncio.Queue[Optional[ChannelEvent]]):
        self.topic = topic
        self._message_queue = queue
        self._eof_received = False

    async def read(self) -> ChannelEvent:
        if self._eof_received:
            raise EOFError("Channel has been closed by peer")

        item = await self._message_queue.get()
        if item is None:
            self._eof_received = True
            raise EOFError("Channel has been closed by peer")
        return item


class NullChannel(Channel):

    async def read(self) -> ChannelEvent:
        raise EOFError()

    async def write(self, item: ChannelEvent):
        pass

    async def write_complete(self):
        pass



async def print_channel_output(channel: ChannelReader, topic_names: Dict[str, str], header: bool = False):
    """
    将Channel内容直接打印到stdout

    :param channel: 监听的 Channel
    :param topic_names: topic 名称和显示名称
    :param header: 是否打印 topic 名称
    """
    current_topic = None
    available_topics = set(topic_names.keys())
    while available_topics:
        try:
            event = await channel.read()
        except EOFError:
            break
        if event.topic not in topic_names:
            continue
        if event.data is None:
            sys.stdout.write('\n')
            available_topics.discard(event.topic)
            continue
        if event.topic != current_topic:
            if header:
                print('===== %s =====' % topic_names[event.topic])
            current_topic = event.topic
        sys.stdout.write(event.data)
        sys.stdout.flush()


class BaseXmlTagFilter(ChannelWriter):
    def __init__(self, tags: Set[str], input_topic: str = 'response'):
        self.tags = tags
        self.input_topic = input_topic
        self.current_tag: Optional[str] = None  # 当前激活的指定标签
        self.current_content: list[str] = []  # 当前标签的内容缓冲区
        self.pending_text: str = ""  # 跨片段的不完整标签缓冲区
        self.pending_closing_tag = ""  # 不完整闭合标签缓冲区

        # 匹配所有XML标签的正则（支持命名空间和特殊字符）
        self.tag_pattern = re.compile(r'<(/?)([a-zA-Z_][\w.:-]*)>')
        # 闭合标签模板（动态生成当前标签的闭合匹配）
        self.closing_tag_template = r'</%s>'

    @abc.abstractmethod
    async def on_tag(self, tag: Optional[str], text: str, end: bool):
        """处理消息片段的标签内容回调"""
        pass

    async def write(self, message: ChannelEvent):
        if message.topic != self.input_topic:
            return
        if message.data is None:
            await self._finalize_fragment()
            return

        full_text = self.pending_text + message.data
        remaining = await self._parse_content(full_text, end=False)
        self.pending_text = remaining

    async def write_complete(self):
        pass

    def _reset_state(self) -> None:
        """重置所有解析状态"""
        self.current_tag = None
        self.current_content = []
        self.pending_text = ""
        self.pending_closing_tag = ""

    async def _parse_content(self, text: str, end: bool) -> str:
        """
        核心解析逻辑：递归处理文本内容
        返回值：未解析的剩余文本（用于跨片段处理）
        """
        pos = 0
        len_text = len(text)

        # 状态1：当前处于指定标签内部（仅搜索当前标签的闭合）
        if self.current_tag is not None:
            closing_tag = self.closing_tag_template % self.current_tag
            closing_len = len(closing_tag)

            # 拼接之前的不完整闭合标签与当前文本
            full_text = self.pending_closing_tag + text
            self.pending_closing_tag = ""

            closing_pos = full_text.find(closing_tag, pos)

            if closing_pos != -1:
                # 找到闭合标签：处理内容并重置状态
                self.current_content.append(full_text[pos:closing_pos])
                await self._emit_content()

                # 继续解析闭合标签后的内容（递归）
                remaining_text = full_text[closing_pos + closing_len:]
                return await self._parse_content(remaining_text, end)
            else:
                # 未找到完整闭合标签：检查是否有部分闭合标签在末尾
                found_partial = False
                # 从最长可能的前缀开始检查，避免误匹配
                for l in range(min(closing_len, len(full_text)), 0, -1):
                    suffix = full_text[-l:]
                    if closing_tag.startswith(suffix):
                        # 保存不完整的闭合标签部分
                        self.pending_closing_tag = suffix
                        # 剩余部分加入内容
                        if len(full_text) > l:
                            self.current_content.append(full_text[:-l])
                        found_partial = True
                        break
                if not found_partial:
                    # 没有部分匹配，全部加入内容
                    self.current_content.append(full_text)
                return ""

        # 状态2：处于顶层（解析所有标签）
        while pos < len_text:
            match = self.tag_pattern.search(text, pos)
            if not match:
                return await self._handle_top_level_remaining(text[pos:], end)

            # 处理标签前的普通文本
            before_tag = text[pos:match.start()]
            if before_tag:
                await self.on_tag(None, before_tag, False)

            # 解析标签信息
            is_closing = match.group(1) == '/'
            tag_name = match.group(2)
            tag_text = match.group(0)

            if tag_name in self.tags:
                if not is_closing:
                    # 处理指定标签的打开：进入标签内部状态
                    self.current_tag = tag_name
                    self.current_content = []
                    pos = match.end()
                    return await self._parse_content(text[pos:], end)
                else:
                    # 孤立的闭合标签：作为普通文本处理
                    await self.on_tag(None, tag_text, False)
                    pos = match.end()
            else:
                # 非指定标签：作为普通文本处理
                await self.on_tag(None, tag_text, False)
                pos = match.end()

        return ""

    async def _handle_top_level_remaining(self, remaining: str, end: bool) -> str:
        """处理顶层未解析的剩余文本（处理不完整标签）"""
        if not remaining:
            return ""

        # 查找最后一个<的位置（判断是否有不完整标签）
        last_less_than = remaining.rfind('<')
        if last_less_than == -1:
            # 无标签结构：全部作为普通文本
            await self.on_tag(None, remaining, end)
            return ""
        else:
            # 分割完整文本与不完整标签
            complete_part = remaining[:last_less_than]
            if complete_part:
                await self.on_tag(None, complete_part, False)
            return remaining[last_less_than:]  # 返回不完整部分

    async def _emit_content(self) -> None:
        """发射当前标签的内容（非空时）"""
        content = ''.join(self.current_content).strip()
        if content:
            await self.on_tag(self.current_tag, content, True)
        self.current_tag = None
        self.current_content = []

    async def _finalize_fragment(self) -> None:
        """处理最后一个片段的未完成状态"""
        if self.current_tag is not None:
            # 处理未闭合的标签内容
            if self.pending_text:
                self.current_content.append(self.pending_text)
            # 加入剩余的不完整闭合标签部分
            if self.pending_closing_tag:
                self.current_content.append(self.pending_closing_tag)
            await self._emit_content()
            self.pending_text = ""
            self.pending_closing_tag = ""  # 新增：清空缓冲区
        elif self.pending_text:
            # 处理未完成的普通文本
            await self.on_tag(None, self.pending_text, True)
            self.pending_text = ""
        else:
            await self.on_tag(None, "", True)


class XmlTagToChannelFilter(BaseXmlTagFilter):
    def __init__(self, output_channel: ChannelWriter, tags: Set[str], input_topic: str = 'response'):
        super().__init__(tags, input_topic)
        self.output_channel = output_channel

    async def on_tag(self, tag: Optional[str], text: str, end: bool):
        """将消息片段分发到对应主题"""
        if text:
            await self.output_channel.write(ChannelEvent(tag, text))
        if end:
            await self.output_channel.write(ChannelEvent(tag, None))

    async def write_complete(self):
        await self.output_channel.write_complete()


class DemuxChannelReader:
    """只读 Channel 包装器，隐藏 write 接口，支持关闭取消订阅"""
    def __init__(self, demux: 'ChannelDemux'):
        self._demux = demux
        self.channel = Channel()
        self._closed = False

    async def read(self) -> ChannelEvent:
        if self._closed:
            raise EOFError("Reader has been closed")
        return await self.channel.read()

    async def close(self):
        """关闭当前reader，取消所有订阅，不再接收新消息"""
        if self._closed:
            return
        self._closed = True
        self._demux.remove_reader(self)
        await self.channel.write_complete()


class ChannelDemux:
    """
    多 topic Channel 拆分与分发器

    使用方式：
        async with ChannelDemux(source_channel) as demux:
            reader1 = demux.get_reader("topic1")
            reader2 = demux.get_reader("topic2", "topic3") # 订阅多个topic
            reader3 = demux.get_reader() # 订阅所有topic
            # 分别使用 reader 并行消费事件
            # 不需要时可调用 await reader.close() 取消订阅
    或使用 start(), close()
    """
    def __init__(self, source: ChannelReader, topics: Optional[Set[str]] = None):
        """
        :param source: 输入 ChannelReader，用于读取 ChannelEvent
        :param topics:  可选的预定义 topic 集合，用于提前初始化订阅列表（不影响动态订阅）
        """
        self._source = source
        self._topics = topics
        self._readers: Dict[str, List[DemuxChannelReader]] = {}          # topic -> List[Channel]
        self._all_subscribers: List[DemuxChannelReader] = []            # 订阅所有topic的通道
        self._reader_topics: Dict[DemuxChannelReader, Optional[Set[str]]] = {}  # reader -> topics (None 表示全部)
        self._dispatch_task: Optional[asyncio.Task] = None
        self._closed = False

        if topics:
            for t in topics:
                self._readers.setdefault(t, [])

    def get_reader(self, *topics: str) -> DemuxChannelReader:
        """
        订阅指定 topic，并返回专属的 DemuxReader。
        不传入topics时订阅所有topic，可同时传入多个topic同时订阅。
        每次调用都会创建一个独立的消费通道，因此多个消费者可以同时订阅同一个 topic 并各自接收完整的事件流。
        调用时机：必须在 demux 上下文内部调用，否则可能抛异常。
        """
        if self._closed:
            raise RuntimeError("ChannelDemux is closed")
        reader = DemuxChannelReader(self)
        topic_set = set(topics)
        if not topic_set:
            # 空参数代表订阅所有topic
            self._all_subscribers.append(reader)
            self._reader_topics[reader] = None
        else:
            for t in topic_set:
                self._readers.setdefault(t, []).append(reader)
            self._reader_topics[reader] = topic_set
        return reader

    def remove_reader(self, reader: DemuxChannelReader):
        """内部方法：根据 reader 实例注销并移除其所有订阅"""
        topics = self._reader_topics.pop(reader, None)
        if topics is None:
            if reader in self._all_subscribers:
                self._all_subscribers.remove(reader)
        else:
            for t in topics:
                if t in self._readers and reader in self._readers[t]:
                    self._readers[t].remove(reader)
                    if not self._readers[t]:
                        del self._readers[t]

    async def _dispatch(self):
        """后台分发协程"""
        try:
            while True:
                event = await self._source.read()
                # 合并对应topic订阅者和全量订阅者
                readers = list(self._readers.get(event.topic, []))
                readers.extend(self._all_subscribers)
                for reader in readers:
                    await reader.channel.write(event)
        except (EOFError, asyncio.CancelledError):
            pass
        finally:
            self._closed = True
            # 通知所有目前活跃的子通道写入完成
            for reader in list(self._reader_topics.keys()):
                await reader.channel.write_complete()

    async def start(self):
        self._dispatch_task = asyncio.create_task(self._dispatch())

    async def close(self):
        if self._dispatch_task:
            self._dispatch_task.cancel()
            try:
                await self._dispatch_task
            except asyncio.CancelledError:
                pass
        self._closed = True

    async def __aenter__(self):
        await self.start()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        await self.close()

